import os
import numpy as np
import xlrd
import matplotlib.pyplot as plt



# task = "large_xyz_xyz_4bits"
# num_bit = 4
# save_path = './scale_qml/save_large/' + task
# num_piece = 4

def plot_save(save_path, ys, name):
    f1 = plt.figure()
    x = np.arange(0, len(ys[0]))
    plt.xlabel('epoch')
    plt.ylabel(name)
    
    plt.plot(x, ys[0], label='training')
    plt.plot(x, ys[1], label='validation')
    plt.legend()

    plt.savefig(save_path + '/{}.jpg'.format(name))



def plot_performance(save_path, task):
    excel_file = os.path.join(save_path, '{}.xls'.format(task))

    wb = xlrd.open_workbook(excel_file)
    sheet = wb.sheet_by_index(0)

    training_acc = np.array(sheet.col_values(2)[1:])
    training_loss = np.array(sheet.col_values(3)[1:])

    val_acc = []
    val_loss = []
    for i in range(1, sheet.nrows):
        if sheet.cell_value(i, 5) != '':
            val_acc.append(sheet.cell_value(i, 5))
            val_loss.append(sheet.cell_value(i, 6))


    intval = len(training_acc) / len(val_acc)
    index = np.arange(0, len(training_acc), step=int(intval))
    training_acc = training_acc[index]
    training_loss = training_loss[index]

    acc_lines = [training_acc,val_acc]
    plot_save(save_path, acc_lines, 'Accuracy')
    loss_lines = [training_loss, val_loss]
    plot_save(save_path, loss_lines, 'Loss')


def plot_gradients(save_path, grad_index):
    gradients_file = os.path.join(save_path, 'gradients_layer.npy')

    grads = np.squeeze(np.load(gradients_file))
    grads = grads[:, :, num_bit:]

    x = np.arange(0, len(grads))
    grad = grads[:, grad_index]

    f1 = plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('{}th gradient value'.format(grad_index))
    
    plt.plot(x, grad)
    # plt.legend()
    plt.savefig(save_path + '/gradient{}.jpg'.format(grad_index))


   


def plot_var_grad(save_path, num_bit, num_piece):
    var_gradients_layer = os.path.join(save_path, 'var_gradients_layer.npy')
    var_gradients_clf = os.path.join(save_path, 'var_gradients_clf.npy')

    vars_pieces = np.load(var_gradients_layer)
    vars_pieces = vars_pieces[:, :, num_bit:]
    vars_clf = np.load(var_gradients_clf)
    vars_clf = vars_clf[:, num_piece:]

    print('--')
    for piece in range(num_piece):
        avg_var = vars_pieces[:, piece, :] 
        avg_var = np.mean(avg_var, 1)
        f1 = plt.figure()
        plt.xlabel('iteration')
        plt.ylabel('avg_Var(gradients)_piece{}'.format(piece))
        x = np.arange(0, len(avg_var))
        plt.plot(x, avg_var)
        plt.savefig(save_path + '/avg_var(gradients)_piece{}.jpg'.format(piece))

    # plot clf var_gradients
    f1 = plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('avg_Var(gradients)_clf')
    avg_var = np.mean(vars_clf, 1)
    x = np.arange(0, len(avg_var))
    plt.plot(x, avg_var)
    plt.savefig(save_path + '/avg_var(gradients)_clf.jpg')


    # vars_pieces = [concate layer and clf]

    # vars = np.squeeze(np.load(gradients_file_layer))
    # grads = grads[:, :, num_bit:]

    # vars = []
    # for i in range(len(grads)):
    #     tmp = []
    #     for j in range(num_piece):
    #         tmp.append(np.var(grads[i, j, :]))          
    #     vars.append(tmp)
    
    # vars = np.array(vars)

    # x = np.arange(0, len(vars))
    # for i in range(num_piece):
    #     f1 = plt.figure()
    #     plt.xlabel('iteration')
    #     plt.ylabel('Var(gradients)')
        
    #     plt.plot(x, vars[:, i])
    #     # plt.legend()
    #     plt.savefig(save_path + '/var(gradients)_{}.jpg'.format(i+1))

    
    # grads = np.squeeze(np.load(gradients_file_clf))
    # grads = grads[:, num_piece:]

    # vars = []
    # for i in range(len(grads)):      
    #     vars.append(np.var(grads[i, :]))
    
    # vars = np.array(vars)

    # x = np.arange(0, len(vars))

    # f1 = plt.figure()
    # plt.xlabel('iteration')
    # plt.ylabel('Var(gradients)')
    
    # plt.plot(x, vars)
    # # plt.legend()
    # plt.savefig(save_path + '/var(gradients)_clf.jpg')



    print('----')








if __name__ == "__main__":
    # plot_performance()
    # plot_gradients(0)
    plot_var_grad()


