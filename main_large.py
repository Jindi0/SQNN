import argparse
import os
from re import X
import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
# from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from util import init_log_cent, dump_circuit
from data_helper import load_raw_data, split_train_validation, shuffle_dataset, img_split
# from callbackfunc import EvalModel_single, GetGradients


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='split_test', help='task name')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--inputsize', type=int, default=8, help='the input size is nxn')
    parser.add_argument('--clfinputsize', type=int, default=2, help='the input size is nxn')
    parser.add_argument('--pieces', type=int, default=4, help='the input size is nxn')
    parser.add_argument('--layers', type=int, default=2, help='the input size is nxn')


    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help="the number of epochs in each global round")
    parser.add_argument('--batchsize', type=int, default=16, help="local batch size")
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


def create_clf_model(inputsize, piece_ind):

    data_qubits = cirq.GridQubit.rect(inputsize, inputsize)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(data_qubits = data_qubits,
                                    readout=readout)


    # Then add layers (experiment by adding more).
    builder.add_input_layer(circuit, cirq.rx, "data{}".format(piece_ind))
    # builder.add_layer(circuit, cirq.XX, "xx{}".format(piece_ind))
    # builder.add_layer(circuit, cirq.YY, "yy{}".format(piece_ind))
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


def create_quantum_model(inputsize, piece_ind):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(inputsize, inputsize)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(data_qubits = data_qubits,
                                    readout=readout)


    # Then add layers (experiment by adding more).
    builder.add_input_layer(circuit, cirq.rx, "data{}".format(piece_ind))
    builder.add_layer(circuit, cirq.XX, "xx{}".format(piece_ind))
    builder.add_layer(circuit, cirq.YY, "yy{}".format(piece_ind))
    # builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)






def main():
    f, sheet = init_log_cent(args)

    save_path = './scale_qml/save_large/' + args.task
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    all_gradients_layer = []
    all_gradients_clf = []
    all_param_layer = []
    all_param_clf = []

    x_train, y_train, x_test, y_test = load_raw_data(args)
    x_train, y_train, x_val, y_val = split_train_validation(x_train, y_train, args.validation_ratio)

    # split data into pieces 
    x_train_pieces, y_train = img_split(args, x_train, y_train)
    x_val_pieces, y_val = img_split(args, x_val, y_val)
    x_test_pieces, y_test = img_split(args, x_test, y_test)


    model_circuits = []
    model_readouts = []
    quantum_layers = []

    for i in range(args.pieces):
        model_circuit, model_readout = create_quantum_model(int(args.inputsize/2), i)
        # model_circuits.append(model_circuit)
        # model_readouts.append(model_readout)
        qlayer = tfq.layers.PQC(model_circuit, model_readout, 
                    initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi, seed=args.seed))
        quantum_layers.append(qlayer)

    clf_circuit, clf_readout = create_clf_model(args.clfinputsize, args.pieces+1)
    clf_layer = tfq.layers.PQC(clf_circuit, clf_readout, 
                initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi, seed=args.seed))

    dump_circuit(model_circuit, dest_path='./scale_qml/save_large/{}/{}.svg'.format(args.task, args.task))
    input_qubits = tfq.convert_to_tensor([cirq.Circuit()])
    
    optimizer = tf.keras.optimizers.SGD(lr=args.lr)
    
    
    # --------------------------------------------------------------------
    tf.config.experimental_run_functions_eagerly(True)

    iterations = int(len(x_train_pieces[0]) / args.batchsize)
    num_data = iterations * args.batchsize

    x_train_pieces = [x[:num_data] for x in x_train_pieces]
    y_train = y_train[:num_data]


    for epoch in range(args.epoch):
        x_train_pieces, y_train = shuffle_dataset(x_train_pieces, y_train)

        for iter in range(iterations):
            x_batch = [x_train[iter*args.batchsize: (iter+1)*args.batchsize] for x_train in x_train_pieces]
            y_batch = y_train[iter*args.batchsize: (iter+1)*args.batchsize]

            batchloss = 0.0
            batch_gradients_layer = []
            batch_gradients_clf = []

            correct_num = 0

            ori_weights = [ql.get_weights()[0] for ql in quantum_layers]
            model_weights = [ow[int(args.inputsize / 2) ** 2:] for ow in ori_weights]

            ori_clf_weight = clf_layer.get_weights()[0]
            clf_weight =ori_clf_weight[args.clfinputsize * args.clfinputsize:]


            for b in range(args.batchsize):
                x = [x_b[b] for x_b in x_batch]
                y = y_batch[b]
                y = 2.0 * y - 1.0

                for cur_piece in range(args.pieces):
                    new_weight = np.concatenate((x[cur_piece].flatten(), model_weights[cur_piece]))
                    quantum_layers[cur_piece].set_weights([new_weight])


                # ============================================= 
                @tf.function()
                def forward():
                    # quantum layer
                    with tf.GradientTape(persistent=True) as tape:
                        outs = []
                        for cur_piece in range(args.pieces):
                            out = quantum_layers[cur_piece](input_qubits)
                            outs.append(out)

                        # set clf parameters
                        outs2input = [o.numpy()[0][0] for o in outs]
                        outs2input = np.array(outs2input)

                        outs2input = np.pi * (outs2input + 1)

                        new_clf_weight = np.concatenate([outs2input, clf_weight])
                        clf_layer.set_weights([new_clf_weight])

                        final_out = clf_layer(input_qubits)
 
                        mse_loss = ((final_out - y) ** 2) / 2

                    dloss_dtheta_clf = tape.gradient(mse_loss, clf_layer.trainable_variables)
                    dout_dtheta = []
                    for cur_piece in range(args.pieces):
                        cur_grad = tape.gradient(outs[cur_piece], quantum_layers[cur_piece].trainable_variables)
                        dout_dtheta.append(cur_grad)

                    del tape

                    return final_out, mse_loss, dout_dtheta, dloss_dtheta_clf
                # ============================================= 
                
                y_pred, loss, dout_dtheta, dloss_dtheta_clf = forward()
                batchloss += loss
                if tf.math.sign(y_pred) == np.sign(y):
                    correct_num += 1

                if dloss_dtheta_clf != None:
                    dloss_dout = dloss_dtheta_clf[0][0:4]
                    dloss_dtheta_layer = []
                    for cur in range(args.pieces):
                        dloss_dtheta_layer_ = dloss_dout[cur] * dout_dtheta[cur] * np.pi
                        dloss_dtheta_layer.append(dloss_dtheta_layer_)
                    
                    batch_gradients_layer.append(dloss_dtheta_layer)
                    batch_gradients_clf.append(dloss_dtheta_clf)

            acc = 100 * correct_num / args.batchsize
            batchloss = batchloss / args.batchsize

            print('Epoch {}, Iteration {}/{}: Loss: {}, Accuracy: {}'.format(epoch, 
                            iter, iterations, batchloss, acc))
       
            batch_gradients_layer0 = tf.squeeze(tf.math.reduce_mean(batch_gradients_layer, 0))
            batch_gradients_clf0 = tf.math.reduce_mean(batch_gradients_clf, 0)

            optimizer.apply_gradients(zip(batch_gradients_clf0, clf_layer.trainable_variables))
            for cur in range(args.pieces):
                tmp = [batch_gradients_layer0[cur]]
                tmp1 = quantum_layers[cur].trainable_variables
                optimizer.apply_gradients(zip(tmp, tmp1))
            

            all_gradients_layer.append(batch_gradients_layer0.numpy())
            all_gradients_clf.append(batch_gradients_clf0.numpy())
            all_param_layer.append([ql.get_weights()[0][:int(args.inputsize / 2) ** 2] for ql in quantum_layers])
            all_param_clf.append(clf_layer.trainable_variables[0].numpy())

            np.save(save_path + '/gradients_layer.npy', np.array(all_gradients_layer))
            np.save(save_path + '/gradients_clf.npy', np.array(all_gradients_clf))
            np.save(save_path + '/all_models_layer.npy', np.array(all_param_layer))
            np.save(save_path + '/all_models_clf.npy', np.array(all_param_clf))

            sheet.write(int(epoch * int(iterations) + iter + 1), 2, acc)
            sheet.write(int(epoch * int(iterations) + iter + 1), 3, float(batchloss.numpy()))
            f.save(save_path + '/{}.xls'.format(args.task))
                
        # validation ---------------------------------------------
        

        ori_weights = [ql.get_weights()[0] for ql in quantum_layers]
        model_weights = [ow[int(args.inputsize / 2) ** 2:] for ow in ori_weights]

        ori_clf_weight = clf_layer.get_weights()[0]
        clf_weight =ori_clf_weight[args.pieces:]

        correct_val = 0
        loss_val = 0.0

        for v in range(len(x_val_pieces)):
            x = x = [x_v[v] for x_v in x_val_pieces]
            y = y_val[v]
            y = 2.0 * y - 1.0

            for cur_piece in range(args.pieces):
                new_weight = np.concatenate((x[cur_piece].flatten(), model_weights[cur_piece]))
                quantum_layers[cur_piece].set_weights([new_weight])

            y_pred, mse_loss, _, _ = forward()
            
            loss_val += mse_loss

            if tf.math.sign(y_pred) == np.sign(y):
                correct_val += 1
        acc_val = 100 * correct_val / len(y_val)
        loss_val = loss_val / len(y_val)
        
        print('Epoch {}, Validation: Loss: {}, Accuracy: {}'.format(epoch, 
                            loss_val, acc_val))

        sheet.write(int(epoch + 1), 5, acc_val)
        sheet.write(int(epoch + 1), 6, float(loss_val.numpy()))
        f.save(save_path + '/{}.xls'.format(args.task))

    
        # Test ---------------------------------------------

        ori_weights = [ql.get_weights()[0] for ql in quantum_layers]
        model_weights = [ow[int(args.inputsize / 2) ** 2:] for ow in ori_weights]

        ori_clf_weight = clf_layer.get_weights()[0]
        clf_weight =ori_clf_weight[args.pieces:]

        correct_test = 0
        loss_test = 0.0

        for v in range(len(x_test_pieces)):
            x = [x_v[v] for x_v in x_test_pieces]
            y = y_test[v]
            y = 2.0 * y - 1.0

            for cur_piece in range(args.pieces):
                new_weight = np.concatenate((x[cur_piece].flatten(), model_weights[cur_piece]))
                quantum_layers[cur_piece].set_weights([new_weight])

            y_pred, mse_loss, _, _ = forward()
            # loss_val += loss

            loss_test += mse_loss

            if tf.math.sign(y_pred) == np.sign(y):
                correct_test += 1
        acc_test = 100 * correct_test / len(y_test)
        loss_test = loss_test / len(y_test)
        
        print('Epoch {}, Testing: Loss: {}, Accuracy: {}'.format(epoch, 
                            loss_test, acc_test))

        sheet.write(int(epoch + 1), 8, acc_test)
        sheet.write(int(epoch + 1), 9, float(loss_test.numpy()))
        f.save(save_path + '/{}.xls'.format(args.task))

        

            


            
    



if __name__ == "__main__":
    main()
    