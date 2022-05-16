import os
import numpy as np
import xlrd
import matplotlib.pyplot as plt

# task = "small_4bits_6gates"
# save_path = './scale_qml/save_small/' + task

# task = "large_xyz_xyz_4bits"
# save_path = './scale_qml/save_large/' + task

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
    grads = grads[:, 4:]

    x = np.arange(0, len(grads))
    grad = grads[:, grad_index]

    f1 = plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('{}th gradient value'.format(grad_index))
    
    plt.plot(x, grad)
    # plt.legend()
    plt.savefig(save_path + '/gradient{}.jpg'.format(grad_index))


   


def plot_var_grad(save_path, args):
    vars_file = os.path.join(save_path, 'var_gradients.npy')

    vars = np.squeeze(np.load(vars_file))
    vars = vars[:, int(args.inputsize * args.inputsize):]

    x = np.arange(0, np.shape(vars)[0])

    # plot var(gradients) for each theta
    for i in range(np.shape(vars)[1]):
        f1 = plt.figure()
        plt.xlabel('iteration')
        plt.ylabel('Var(gradients{})'.format(i))
        plt.plot(x, vars[:, i])
        plt.savefig(save_path + '/var(gradients{}).jpg'.format(i))

    # plot the average of var(gradients)
    vars = np.mean(vars, 1)
    f1 = plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('avg Var(gradients)')
    plt.plot(x, vars)
    plt.savefig(save_path + '/avg_var(gradients).jpg')


    print('----')








if __name__ == "__main__":
    task = "small_3gates_4bits"
    save_path = './scale_qml/save_small/' + task
    # plot_performance()
    plot_gradients(save_path, 4)
    # plot_var_grad()


