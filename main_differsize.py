import argparse
import os
from re import X
import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
from util import init_log, dump_circuit
from data_helper import load_raw_data, split_train_validation, img_split_3piece
from sklearn.utils import shuffle


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='sqnn_differsize', help='task name')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--inputsize', type=int, default=4, help='the input size is nxn')
    parser.add_argument('--clfinputsize', type=int, default=3, help='the input size of quantum predictor is n')
    parser.add_argument('--pieces', type=int, default=3, help='the number of segments')


    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help="the number of epochs in each global round")
    parser.add_argument('--batchsize', type=int, default=32, help="local batch size")
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
    '''
        Create the circuit of quantum predictor for classification
    '''
    data_qubits = cirq.GridQubit.rect(1, inputsize)  # a 1xinputsize grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(data_qubits = data_qubits,
                                    readout=readout)

    # add input layer for angle encoding
    builder.add_input_layer(circuit, cirq.rx, "data{}".format(piece_ind))

    # add variational quantum layer with Ising coupling gates
    # builder.add_layer(circuit, cirq.XX, "xx{}".format(piece_ind))
    builder.add_layer(circuit, cirq.YY, "yy{}".format(piece_ind))
    # builder.add_layer(circuit, cirq.YY, "zz{}".format(piece_ind))

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


def create_quantum_model(piecesize, piece_ind):
    '''
        Create the circuit for quantum feature extractor
    '''
    data_qubits = cirq.GridQubit.rect(1, piecesize)  # a 1xpiecesize grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(data_qubits = data_qubits,
                                    readout=readout)


    # add input layer for angle encoding
    builder.add_input_layer(circuit, cirq.rx, "data{}".format(piece_ind))

    # add variational quantum layer with Ising coupling gates
    builder.add_layer(circuit, cirq.XX, "xx{}".format(piece_ind))
    builder.add_layer(circuit, cirq.YY, "yy{}".format(piece_ind))
    builder.add_layer(circuit, cirq.YY, "zz{}".format(piece_ind))

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)

def shuffle_dataset(x, y):
    x0, x1, x2, y = shuffle(x[0], x[1], x[2], y)
    return [x0, x1, x2], y

def evaluate_model(quantum_layers, clf_layer, piecesizes, eval_x, eval_y, epoch, phase, sheet, f, save_path):
    input_qubits = tfq.convert_to_tensor([cirq.Circuit()])

    ori_weights = [ql.get_weights()[0] for ql in quantum_layers]
    model_weights = [ori_weights[i][piecesizes[i]:] for i in range(args.pieces)]

    ori_clf_weight = clf_layer.get_weights()[0]
    clf_weight =ori_clf_weight[args.pieces:]

    correct_num = 0
    loss = 0.0

    for v in range(len(eval_y)):
        x = [x_v[v] for x_v in eval_x]
        y = eval_y[v]
        y = 2.0 * y - 1.0

        for cur_piece in range(args.pieces):
            new_weight = np.concatenate((x[cur_piece].flatten(), model_weights[cur_piece]))
            quantum_layers[cur_piece].set_weights([new_weight])

        # ============================================= 
        @tf.function()
        def prediction():
            # quantum layer
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

            return final_out, mse_loss
        # ============================================= 
        y_pred, mse_loss = prediction()
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
    small_segment_size = int(args.inputsize / 2)**2
    piecesizes = [small_segment_size * 2, small_segment_size, small_segment_size]
    f, sheet = init_log(args)
    if not os.path.exists('./scale_qml/save_differsize/'):
        os.mkdir('./scale_qml/save_differsize/')

    save_path = './scale_qml/save_differsize/' + args.task
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    x_train, y_train, x_test, y_test = load_raw_data(args)
    x_train, y_train, x_val, y_val = split_train_validation(x_train, y_train, args.validation_ratio)

    # split data into pieces 
    x_train_pieces, y_train = img_split_3piece(args, x_train, y_train)
    x_val_pieces, y_val = img_split_3piece(args, x_val, y_val)
    x_test_pieces, y_test = img_split_3piece(args, x_test, y_test)


    quantum_layers = []

    for i in range(args.pieces):
        model_circuit, model_readout = create_quantum_model(piecesizes[i], i)
        qlayer = tfq.layers.PQC(model_circuit, model_readout, 
                    initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi, seed=args.seed))
        quantum_layers.append(qlayer)

    clf_circuit, clf_readout = create_clf_model(args.clfinputsize, args.pieces+1)
    clf_layer = tfq.layers.PQC(clf_circuit, clf_readout, 
                initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi, seed=args.seed))

    dump_circuit(model_circuit, dest_path='./scale_qml/save_differsize/{}/{}.svg'.format(args.task, args.task))
    input_qubits = tfq.convert_to_tensor([cirq.Circuit()])
    
    optimizer = tf.keras.optimizers.SGD(lr=args.lr)

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
            model_weights = [ori_weights[i][piecesizes[i]:] for i in range(args.pieces)]

            ori_clf_weight = clf_layer.get_weights()[0]
            clf_weight =ori_clf_weight[args.pieces:]


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
                    dloss_dout = dloss_dtheta_clf[0][0:args.pieces]
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

            batch_gradients_layer_1 = []
            batch_gradients_layer_2 = []
            batch_gradients_layer_3 = []
            

            for j in range(args.batchsize):
                batch_gradients_layer_1.append(batch_gradients_layer[j][0])
                batch_gradients_layer_2.append(batch_gradients_layer[j][1])
                batch_gradients_layer_3.append(batch_gradients_layer[j][2])

            batch_gradients_layer_1 = tf.squeeze(tf.math.reduce_mean(batch_gradients_layer_1, 0))
            batch_gradients_layer_2 = tf.squeeze(tf.math.reduce_mean(batch_gradients_layer_2, 0))
            batch_gradients_layer_3 = tf.squeeze(tf.math.reduce_mean(batch_gradients_layer_3, 0))
            batch_gradients_layer0 = [batch_gradients_layer_1, batch_gradients_layer_2, batch_gradients_layer_3]

            batch_gradients_clf0 = tf.math.reduce_mean(batch_gradients_clf, 0)

            optimizer.apply_gradients(zip(batch_gradients_clf0, clf_layer.trainable_variables))
            for cur in range(args.pieces):
                tmp = [batch_gradients_layer0[cur]]
                tmp1 = quantum_layers[cur].trainable_variables
                optimizer.apply_gradients(zip(tmp, tmp1))

            sheet.write(int(epoch * int(iterations) + iter + 1), 2, acc)
            sheet.write(int(epoch * int(iterations) + iter + 1), 3, float(batchloss.numpy()))
            f.save(save_path + '/{}.xls'.format(args.task))
                
        # ----------- Evaluation -------------------------------------------
        evaluate_model(quantum_layers, clf_layer, piecesizes, x_val_pieces, y_val, epoch, 'val', sheet, f, save_path)
        evaluate_model(quantum_layers, clf_layer, piecesizes, x_test_pieces, y_test, epoch, 'test', sheet, f, save_path)
        
        

        
   

if __name__ == "__main__":
    main()
    