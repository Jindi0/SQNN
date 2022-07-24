import argparse
import os
from re import X
import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from util import init_log, dump_circuit
from data_helper import load_raw_data, split_train_validation
from results_analy_small import plot_performance, plot_var_grad
# from callbackfunc import EvalModel_single, GetGradients


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='qnn_4qb', help='task name')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--inputsize', type=int, default=2, help='the input size is nxn')

    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help="the number of epochs")
    parser.add_argument('--batchsize', type=int, default=32, help="the number of instances in a batch")
    parser.add_argument('--validation_ratio', type=float, default=0.2, help='the ratio of validation dataset')

    args = parser.parse_args()
    return args

args = args_parser()




class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_input_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i).zfill(2))
            circuit.append(gate(symbol)(qubit))

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i).zfill(2))
            circuit.append(gate(qubit, self.readout)**symbol)



def create_quantum_model(inputsize):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(inputsize, inputsize)  # a inputsizexinputsize grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(data_qubits = data_qubits,
                                    readout=readout)


    # add an input layer for encoding classical data into quantum data 
    # with angle encoding method
    builder.add_input_layer(circuit, cirq.rx, "data0")

    # add variational quantum layer with Ising coupling gates
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.YY, "yy1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    builder.add_layer(circuit, cirq.XX, "xx2")
    builder.add_layer(circuit, cirq.YY, "yy2")
    builder.add_layer(circuit, cirq.ZZ, "zz2")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


def evaluate_model(modelqlayer, eval_x, eval_y, epoch, phase, sheet, f, save_path):
    ori_weights = modelqlayer.get_weights()[0]
    model_weights = ori_weights[args.inputsize * args.inputsize:]

    input_qubits = tfq.convert_to_tensor([cirq.Circuit()])

    correct_num = 0
    loss = 0.0

    for v in range(len(eval_x)):
        x = eval_x[v]
        y = eval_y[v]
        y = 2.0 * y - 1.0

        new_weights = np.concatenate((x.flatten(), model_weights))
        modelqlayer.set_weights([new_weights])

        y_pred = modelqlayer(input_qubits)
        mse_loss = ((y_pred - y) ** 2) / 2
        loss += mse_loss

        if tf.math.sign(y_pred) == np.sign(y):
            correct_num += 1
    acc_eval = 100 * correct_num / len(eval_y)
    loss_eval = loss / len(eval_y)
    

    if phase == 'val':
        sheet.write(int(epoch + 1), 5, acc_eval)
        sheet.write(int(epoch + 1), 6, float(loss_eval.numpy()))
        print('Epoch {}, Validation: Loss: {}, Accuracy: {}'.format(epoch, 
                        loss_eval, acc_eval))
    elif phase == 'test':
        sheet.write(int(epoch + 1), 8, acc_eval)
        sheet.write(int(epoch + 1), 9, float(loss_eval.numpy()))
        print('Epoch {}, Test: Loss: {}, Accuracy: {}'.format(epoch, 
                        loss_eval, acc_eval))
    f.save(save_path + '/{}.xls'.format(args.task))




def main():
    f, sheet = init_log(args)
    if not os.path.exists('./scale_qml/save_qnn/'):
        os.mkdir('./scale_qml/save_qnn/')

    save_path = './scale_qml/save_qnn/' + args.task
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    x_train, y_train, x_test, y_test = load_raw_data(args)
    x_train, y_train, x_val, y_val = split_train_validation(x_train, y_train, args.validation_ratio)
   

    model_circuit, model_readout = create_quantum_model(args.inputsize)

    # draw the created circuit
    dump_circuit(model_circuit, dest_path='./scale_qml/save_qnn/{}/{}.svg'.format(args.task, args.task))

    input_qubits = tfq.convert_to_tensor([cirq.Circuit()])
    modelqlayer = tfq.layers.PQC(model_circuit, model_readout, 
                    initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi, seed=args.seed))

    optimizer = tf.keras.optimizers.SGD(lr=args.lr)
    tf.config.experimental_run_functions_eagerly(True)

    iterations = int(len(x_train) / args.batchsize)
    num_data = iterations * args.batchsize

    x_train = x_train[:num_data]
    y_train = y_train[:num_data]


    for epoch in range(args.epoch):
        x_train, y_train = shuffle(x_train, y_train)

        for iter in range(iterations):
            x_batch = x_train[iter*args.batchsize: (iter+1)*args.batchsize]
            y_batch = y_train[iter*args.batchsize: (iter+1)*args.batchsize]

            batchloss = 0.0
            batch_gradients = []
            correct_num = 0

            
            ori_weights = modelqlayer.get_weights()[0]
            model_weights = ori_weights[args.inputsize * args.inputsize:]

            for b in range(args.batchsize):
                x = x_batch[b]
                y = y_batch[b]
                y = 2.0 * y - 1.0

                # load a training instance in QNN input layer
                new_weights = np.concatenate((x.flatten(), model_weights))
                modelqlayer.set_weights([new_weights])

                # ============================================= 
                @tf.function()
                def forward():
                    # quantum layer
                    with tf.GradientTape() as tape:
                        out = modelqlayer(input_qubits)
                        mse_loss = ((out - y) ** 2) / 2
                    dloss_dtheta = tape.gradient(mse_loss, modelqlayer.trainable_variables)

                    return out, mse_loss, dloss_dtheta
                # ============================================= 
                
                y_pred, loss, dloss_dtheta = forward()
                batchloss += loss
                if tf.math.sign(y_pred) == np.sign(y):
                    correct_num += 1

                if dloss_dtheta != None:
                    batch_gradients.append(dloss_dtheta)

            acc = 100 * correct_num / args.batchsize
            batchloss = batchloss / args.batchsize

            print('Epoch {}, Iteration {}/{}: Loss: {}, Accuracy: {}'.format(epoch, 
                            iter, iterations, batchloss, acc))

            batch_gradients = tf.squeeze(tf.stack(batch_gradients))
            batch_gradients0 = tf.math.reduce_mean(batch_gradients, 0)
            optimizer.apply_gradients(zip([batch_gradients0], modelqlayer.trainable_variables))

            sheet.write(int(epoch * int(iterations) + iter + 1), 2, acc)
            sheet.write(int(epoch * int(iterations) + iter + 1), 3, float(batchloss.numpy()))
            f.save(save_path + '/{}.xls'.format(args.task))
                
        # ----------- Evaluation ---------------------------------------------
        evaluate_model(modelqlayer, x_val, y_val, epoch, 'val', sheet, f, save_path)
        evaluate_model(modelqlayer, x_test, y_test, epoch, 'test', sheet, f, save_path)

        

    


if __name__ == "__main__":
    main()
    
    