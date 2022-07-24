import argparse
import os
from re import X
import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import statistics
import matplotlib.pyplot as plt
from util import init_log_local, dump_circuit
from data_helper import load_raw_data, split_train_validation, shuffle_dataset, img_split
# from callbackfunc import EvalModel_single, GetGradients
from results_analy_large import plot_performance, plot_var_grad


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='xyz_y_30_50_4bit', help='task name')
    parser.add_argument('--note', type=str, default='-', help='task name')

    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--inputsize', type=int, default=4, help='the input size is nxn')
    parser.add_argument('--clfinputsize', type=int, default=2, help='the input size is nxn')
    parser.add_argument('--pieces', type=int, default=4, help='the input size is nxn')

    parser.add_argument('--local-epochs', type=int, default=25, help='the input size is nxn')
    parser.add_argument('--remote-epochs', type=int, default=50, help='the input size is nxn')

    # parser.add_argument('--layers', type=int, default=2, help='the input size is nxn')


    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--remote_lr', type=float, default=0.01, help='learning rate')
    # parser.add_argument('--epoch', type=int, default=100, help="the number of epochs in each global round")
    parser.add_argument('--batchsize', type=int, default=32, help="local batch size")
    parser.add_argument('--validation_ratio', type=float, default=0, help='the ratio of validation dataset')

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
    builder.add_layer(circuit, cirq.YY, "yy{}".format(piece_ind))
    # builder.add_layer(circuit, cirq.YY, "zz{}".format(piece_ind))
    # builder.add_layer(circuit, cirq.XX, "xx1{}".format(piece_ind))
    # builder.add_layer(circuit, cirq.YY, "yy1{}".format(piece_ind))
    # builder.add_layer(circuit, cirq.YY, "zz1{}".format(piece_ind))

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


def create_quantum_model(inputsize, piece_ind):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(inputsize, inputsize)  
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
    builder.add_layer(circuit, cirq.YY, "zz{}".format(piece_ind))
    # builder.add_layer(circuit, cirq.XX, "xx1{}".format(piece_ind))
    # builder.add_layer(circuit, cirq.YY, "yy1{}".format(piece_ind))
    # builder.add_layer(circuit, cirq.YY, "zz1{}".format(piece_ind))

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)




def test_local(epoch, x_test_pieces, y_test, quantum_layers, input_qubits, f, sheet, save_path):
    ori_weights = [ql.get_weights()[0] for ql in quantum_layers]
    model_weights = [ow[int(args.inputsize / 2) ** 2:] for ow in ori_weights]

    correct_val = [0 for i in range(args.pieces)]
    loss_val = [0 for i in range(args.pieces)]


    for v in range(len(y_test)):
        x = [x_v[v] for x_v in x_test_pieces]
        y = y_test[v]
        y = 2.0 * y - 1.0

        for cur_piece in range(args.pieces):
            new_weight = np.concatenate((x[cur_piece].flatten(), model_weights[cur_piece]))
            quantum_layers[cur_piece].set_weights([new_weight])
        
        # ============================================= 
        def prediction():
            outs = []
            localloss = []

            for cur_piece in range(args.pieces):
                out = quantum_layers[cur_piece](input_qubits)
                outs.append(out)
                localloss.append(((out - y) ** 2) / 2)
            return outs, localloss
        # ============================================= 

        y_pred, mse_loss = prediction()

        for cur_piece in range(args.pieces):
            loss_val[cur_piece] += mse_loss[cur_piece].numpy()[0][0]
            if tf.math.sign(y_pred[cur_piece]) == np.sign(y):
                correct_val[cur_piece] += 1

    accs = []
    for cur_piece in range(args.pieces):
        loss_val[cur_piece] /= len(y_test)
        acc = 100. * correct_val[cur_piece] / len(y_test)
        accs.append(acc)

        sheet.write(epoch, 11+(2*cur_piece), accs[cur_piece])
        sheet.write(epoch, 12+(2*cur_piece), loss_val[cur_piece])

    f.save(save_path + '/{}.xls'.format(args.task))
    print('\nTest Local set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        statistics.mean(loss_val), statistics.mean(correct_val), len(y_test), statistics.mean(accs)))



def test_remote(epoch, x_test_pieces, y_test, quantum_layers, clf_layer, input_qubits, f, sheet, save_path):
    ori_weights = [ql.get_weights()[0] for ql in quantum_layers]
    model_weights = [ow[int(args.inputsize / 2) ** 2:] for ow in ori_weights]

    ori_clf_weight = clf_layer.get_weights()[0]
    clf_weight =ori_clf_weight[args.clfinputsize * args.clfinputsize:]

    correct_val = 0
    loss_val = 0

    for v in range(len(y_test)):
        x = [x_v[v] for x_v in x_test_pieces]
        y = y_test[v]
        y = 2.0 * y - 1.0

        for cur_piece in range(args.pieces):
            new_weight = np.concatenate((x[cur_piece].flatten(), model_weights[cur_piece]))
            quantum_layers[cur_piece].set_weights([new_weight])

        # ============================================= 
        def prediction_clf():
            outs = []
            for cur_piece in range(args.pieces):
                out = quantum_layers[cur_piece](input_qubits)
                outs.append(out)
            outs2input = [o.numpy()[0][0] for o in outs]
            outs2input = np.array(outs2input)
            outs2input = np.pi * (outs2input + 1)
            new_clf_weight = np.concatenate([outs2input, clf_weight])
            clf_layer.set_weights([new_clf_weight])
            final_out = clf_layer(input_qubits)
            mse_loss = ((final_out - y) ** 2) / 2
            return final_out, mse_loss
        # ============================================= 
        y_pred, loss = prediction_clf()
        loss_val += loss.numpy()[0][0]
        if tf.math.sign(y_pred) == np.sign(y):
            correct_val += 1

    acc = 100. * correct_val / len(y_test)
    test_loss = loss_val / len(y_test)

    sheet.write(epoch, 23, acc)
    sheet.write(epoch, 24, test_loss)
    f.save(save_path + '/{}.xls'.format(args.task))


    









def main():
    f, sheet = init_log_local(args)

    save_path = './scale_qml/save_local/' + args.task
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    all_gradients_layer = []
    all_gradients_clf = []
    all_param_layer = []
    all_param_clf = []
    var_gradients_clf = []
    var_gradients_layers = []

    x_train, y_train, x_test, y_test = load_raw_data(args)
    x_train, y_train, x_val, y_val = split_train_validation(x_train, y_train, args.validation_ratio)

    # split data into pieces 
    x_train_pieces, y_train = img_split(args, x_train, y_train)
    # x_val_pieces, y_val = img_split(args, x_val, y_val)
    x_test_pieces, y_test = img_split(args, x_test, y_test)


    quantum_layers = []

    for i in range(args.pieces):
        model_circuit, model_readout = create_quantum_model(int(args.inputsize/2), i)
        qlayer = tfq.layers.PQC(model_circuit, model_readout, 
                    initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi, seed=args.seed))
        quantum_layers.append(qlayer)

    clf_circuit, clf_readout = create_clf_model(args.clfinputsize, args.pieces+1)
    clf_layer = tfq.layers.PQC(clf_circuit, clf_readout, 
                initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi, seed=args.seed))

    # dump_circuit(model_circuit, dest_path='./scale_qml/save_local/{}/{}.svg'.format(args.task, args.task))
    
    # the qubits for loading data
    input_qubits = tfq.convert_to_tensor([cirq.Circuit()])
    
    optimizer = tf.keras.optimizers.SGD(lr=args.lr)
    
    
    # --------------------------------------------------------------------
    tf.config.run_functions_eagerly(True)

    iterations = int(len(x_train_pieces[0]) / args.batchsize)
    num_data = iterations * args.batchsize

    x_train_pieces = [x[:num_data] for x in x_train_pieces]
    y_train = y_train[:num_data]

    # local training
    for epoch in range(args.local_epochs):
        x_train_pieces, y_train = shuffle_dataset(x_train_pieces, y_train)
        for iter in range(iterations):
            x_batch = [x_train[iter*args.batchsize: (iter+1)*args.batchsize] for x_train in x_train_pieces]
            y_batch = y_train[iter*args.batchsize: (iter+1)*args.batchsize]

            # metrix initialization
            local_batchloss = [0.0 for i in range(args.pieces)]
            correct_num = [0.0 for i in range(args.pieces)]

            batch_gradients_layer = []

            ori_weights = [ql.get_weights()[0] for ql in quantum_layers]
            model_weights = [ow[int(args.inputsize / 2) ** 2:] for ow in ori_weights]

            for b in range(args.batchsize):
                x = [x_b[b] for x_b in x_batch]
                y = y_batch[b]
                y = 2.0 * y - 1.0

                for cur_piece in range(args.pieces):
                    new_weight = np.concatenate((x[cur_piece].flatten(), model_weights[cur_piece]))
                    quantum_layers[cur_piece].set_weights([new_weight])

                # ============================================= 
                @tf.function()
                def forward_local():
                    # quantum layer
                    with tf.GradientTape(persistent=True) as tape:
                        outs = []
                        localloss = []
                        for cur_piece in range(args.pieces):
                            out = quantum_layers[cur_piece](input_qubits)
                            outs.append(out)
                            localloss.append(((out - y) ** 2) / 2)

                    dlocalloss_dtheta = []
                    for cur_piece in range(args.pieces):
                        cur_grad = tape.gradient(localloss[cur_piece], quantum_layers[cur_piece].trainable_variables)
                        dlocalloss_dtheta.append(cur_grad)

                    del tape
                    return outs, localloss, dlocalloss_dtheta
                # ============================================= 

                local_outs, local_loss, dlocalloss_dtheta = forward_local()
                for cur_piece in range(args.pieces):
                    local_batchloss[cur_piece] += local_loss[cur_piece].numpy()[0]
                    if tf.math.sign(local_outs[cur_piece]) == np.sign(y):
                        correct_num[cur_piece] += 1

                batch_gradients_layer.append(dlocalloss_dtheta)
            accs = [0.0, 0.0, 0.0, 0.0]
            tmp_batchloss = [0.0, 0.0, 0.0, 0.0]
            for cur_piece in range(args.pieces):
                accs[cur_piece] = 100 * correct_num[cur_piece] / args.batchsize
                tmp_batchloss[cur_piece] = local_batchloss[cur_piece][0] / args.batchsize

            print('Epoch {}-{}, Iteration {}/{}: (Accuracy, Loss): ({}%, {:.2f}), ({}%, {:.2f}), ({}%, {:.2f}), ({}%, {:.2f})'.format(
                epoch, epoch, iter, iterations, 
                accs[0], tmp_batchloss[0], accs[1], tmp_batchloss[1], 
                accs[2], tmp_batchloss[2], accs[3], tmp_batchloss[3])) 

            batch_gradients_layer0 = tf.squeeze(tf.math.reduce_mean(batch_gradients_layer, 0))
            for cur in range(args.pieces):
                tmp = [batch_gradients_layer0[cur]]
                tmp1 = quantum_layers[cur].trainable_variables
                optimizer.apply_gradients(zip(tmp, tmp1))

                sheet.write(int(epoch * int(iterations) + iter + 1), 2 + 2*cur, accs[cur])
                sheet.write(int(epoch * int(iterations) + iter + 1), 3 + 2*cur, tmp_batchloss[cur])
            f.save(save_path + '/{}.xls'.format(args.task))


        # test local models
        # validation ---------------------------------------------
        test_local(epoch, x_test_pieces, y_test, quantum_layers, input_qubits, f, sheet, save_path)

#===================================



    # remote trianing
    optimizer = tf.keras.optimizers.SGD(lr=args.remote_lr)

    for epoch in range(args.remote_epochs):
        x_train_pieces, y_train = shuffle_dataset(x_train_pieces, y_train)

       
        for iter in range(iterations):
            x_batch = [x_train[iter*args.batchsize: (iter+1)*args.batchsize] for x_train in x_train_pieces]
            y_batch = y_train[iter*args.batchsize: (iter+1)*args.batchsize]

            batchloss = 0.0
            batch_gradients_clf = []
            correct_num_clf = 0

            ori_weights = [ql.get_weights()[0] for ql in quantum_layers]
            model_weights = [ow[int(args.inputsize / 2) ** 2:] for ow in ori_weights]

            ori_clf_weight = clf_layer.get_weights()[0]
            clf_weight =ori_clf_weight[args.clfinputsize * args.clfinputsize:]

            for b in range(args.batchsize):
                x = [x_b[b] for x_b in x_batch]
                y = y_batch[b]
                y = 2.0 * y - 1.0

                # set extractor parameters
                for cur_piece in range(args.pieces):
                    new_weight = np.concatenate((x[cur_piece].flatten(), model_weights[cur_piece]))
                    quantum_layers[cur_piece].set_weights([new_weight])

                # ============================================= 
                @tf.function()
                def forward_clf():
                    outs = []
                    for cur_piece in range(args.pieces):
                        out = quantum_layers[cur_piece](input_qubits)
                        outs.append(out)
                    outs2input = [o.numpy()[0][0] for o in outs]
                    outs2input = np.array(outs2input)

                    with tf.GradientTape() as tape:
                        # set clf parameters
                        outs2input = np.pi * (outs2input + 1)
                        new_clf_weight = np.concatenate([outs2input, clf_weight])
                        clf_layer.set_weights([new_clf_weight])

                        final_out = clf_layer(input_qubits)
                        mse_loss = ((final_out - y) ** 2) / 2

                    dloss_dtheta_clf = tape.gradient(mse_loss, clf_layer.trainable_variables)
                   
                    return final_out, mse_loss, dloss_dtheta_clf
                # ============================================= 
                y_pred, loss, dloss_dtheta_clf = forward_clf()
                batchloss += loss
                if tf.math.sign(y_pred) == np.sign(y):
                    correct_num_clf += 1
                batch_gradients_clf.append(dloss_dtheta_clf)
            acc = 100 * correct_num_clf / args.batchsize
            batchloss = batchloss[0]  / args.batchsize
            print('Epoch {}, Iteration {}/{}: Accuracy: {}, Loss: {}'.format(epoch, 
                            iter, iterations, acc, batchloss))

            batch_gradients_clf0 = tf.math.reduce_mean(batch_gradients_clf, 0)
            optimizer.apply_gradients(zip(batch_gradients_clf0, clf_layer.trainable_variables))
            
            sheet.write(int(epoch * int(iterations) + iter + 1), 21, acc)
            sheet.write(int(epoch * int(iterations) + iter + 1), 22, float(batchloss.numpy()[0]))
            f.save(save_path + '/{}.xls'.format(args.task))

        # test remote models
        # validation ---------------------------------------------
        test_remote(epoch, x_test_pieces, y_test, quantum_layers, clf_layer, input_qubits, f, sheet, save_path)
    


# ---------------------------------------------------------------------------



   
    



if __name__ == "__main__":
    main()
    # save_path = './scale_qml/save_local/' + args.task
    # plot_performance(save_path, args.task)
    # plot_var_grad(save_path, int((args.inputsize / 2) ** 2), args.pieces)
    