import os
import numpy as np
import xlrd
import matplotlib.pyplot as plt
# from scipy import interpolate
from scipy.interpolate import make_interp_spline
from scipy import interpolate

figurepath = './scale_qml/new_figures/' 

def extend(valuelists):
    length = max([len(values) for values in valuelists])
    new_lists = []

    for values in valuelists:
        if len(values) < length:
            values = values.tolist()
            num = length - len(values)
            last = values[-1]
            for i in range(num):
                values.append(last)
        values = np.array(values)

        new_lists.append(values)

    return new_lists





def plot_trainloss(savepath, tasks, names):
    excel_files = [os.path.join(savepath, task, '{}.xls'.format(task)) for task in tasks]
    losses = []
    for file in excel_files:
        wb = xlrd.open_workbook(file)
        sheet = wb.sheet_by_index(0)
        values = np.array(sheet.col_values(3)[1:])
        intval = len(values) / 100
        index = np.arange(0, len(values), step=int(intval))
        values = values[index]

        losses.append(values)

    losses = extend(losses)

    f1 = plt.figure()
    x = np.arange(0, len(losses[0]))

    plt.xlabel('epochs')
    plt.ylabel('Training loss')

    for i in range(len(losses)):
        power_smooth = losses[i]

        plt.plot(x, power_smooth, lw=1.5, label=names[i])
    
    plt.legend(loc = 'upper right')
    plt.savefig(figurepath + '/large_loss_4.pdf', dpi=1200)



# def plot_trainacc(savepath, tasks, names):
#     excel_files = [os.path.join(savepath, task, '{}.xls'.format(task)) for task in tasks]
#     losses = []
#     for file in excel_files:
#         wb = xlrd.open_workbook(file)
#         sheet = wb.sheet_by_index(0)
#         values = np.array(sheet.col_values(2)[1:])
#         intval = len(values) / 100
#         index = np.arange(0, len(values), step=int(intval))
#         values = values[index]

#         losses.append(values)

#     losses = extend(losses)

#     f1 = plt.figure()
#     x = np.arange(0, len(losses[0]))

#     plt.xlabel('epochs')
#     plt.ylabel('Training accuracy')

#     for i in range(len(losses)):
#         power_smooth = losses[i]

#         plt.plot(x, power_smooth, lw=1.5, label=names[i])
    
#     plt.legend(loc = 'lower right')
#     plt.savefig(figurepath + '/large_trainingacc_6g.png', dpi=1200)



def plot_valacc(savepath, tasks, names, maxacc):
    excel_files = [os.path.join(savepath, task, '{}.xls'.format(task)) for task in tasks]
    accs = []
    for file in excel_files:
        wb = xlrd.open_workbook(file)
        sheet = wb.sheet_by_index(0)
        val_acc = []
        for i in range(1, sheet.nrows):
            if sheet.cell_value(i, 5) != '':
                val_acc.append(sheet.cell_value(i, 5))

        accs.append(val_acc)

    # losses = extend(losses)
    f1 = plt.figure()
    x = np.arange(0, len(accs[0]))

    plt.xlabel('epochs')
    plt.ylabel('Validation accuracy')

    for i in range(len(accs)):
        power_smooth = accs[i]

        plt.plot(x, power_smooth, lw=1.5, label=names[i])
        plt.plot(maxacc[i][0], maxacc[i][1], marker='o', color='r')
    
    plt.legend(loc = 'lower right')
    plt.savefig(figurepath + '/large_valacc_4.pdf', dpi=1200)



# def plot_testacc(savepath, tasks, names):
#     excel_files = [os.path.join(savepath, task, '{}.xls'.format(task)) for task in tasks]
#     accs = []
#     for file in excel_files:
#         wb = xlrd.open_workbook(file)
#         sheet = wb.sheet_by_index(0)
#         val_acc = []
#         for i in range(1, sheet.nrows):
#             if sheet.cell_value(i, 8) != '':
#                 val_acc.append(sheet.cell_value(i, 8))

#         accs.append(val_acc)

#     # losses = extend(losses)
#     f1 = plt.figure()
#     x = np.arange(0, len(accs[0]))

#     plt.xlabel('epochs')
#     plt.ylabel('Test accuracy')

#     for i in range(len(accs)):
#         power_smooth = accs[i]

#         plt.plot(x, power_smooth, lw=1.5, label=names[i])
    
#     plt.legend(loc = 'lower right')
#     plt.savefig(figurepath + '/small_testacc_6g.png', dpi=1200)


        
def compare_gates():
    save_path_small = './scale_qml/save_small/'
    save_path_large = './scale_qml/save_large/'
    # tasks = ["small_3gates_16bits", 'small_6gates_16bits']
    names = ['17qb_3layer_qnn', '5qb_sqnn', 'uneven_sqnn']
    task1 = 'small_3gates_16bits'
    task2 = 'large_xyz_y_4bits'
    task3 = 'diffsize_xyz_y_844'

    # excel_files = [os.path.join(save_path_small, task, '{}.xls'.format(task)) for task in tasks]
    path1 = os.path.join(save_path_small, task1, '{}.xls'.format(task1))
    path2 = os.path.join(save_path_large, task2, '{}.xls'.format(task2))
    path3 = os.path.join(save_path_large, task3, '{}.xls'.format(task3))

    excel_files = [path1, path2, path3]

    # maxacc = [(31, 92.042042), (10, 92.5925926)]

    accs = []
    for file in excel_files:
        wb = xlrd.open_workbook(file)
        sheet = wb.sheet_by_index(0)
        val_acc = []
        for i in range(1, sheet.nrows):
            if sheet.cell_value(i, 5) != '':
                val_acc.append(sheet.cell_value(i, 5))

        accs.append(val_acc)

    f1 = plt.figure()
    x = np.arange(0, len(accs[0]))
    plt.xlabel('epochs')
    plt.ylabel('Validation accuracy')

    for i in range(len(accs)):
        plt.plot(x, accs[i], lw=1.5, label=names[i])
        # plt.plot(maxacc[i][0], maxacc[i][1], marker='o', color='r')

    plt.legend(loc = 'lower right')
    plt.savefig(figurepath + '/comp_differsize.pdf', dpi=1200)



def compare_loss():
    save_path_small = './scale_qml/save_small/'
    save_path_large = './scale_qml/save_large/'
    # tasks = ["small_3gates_16bits", 'small_6gates_16bits']
    names = ['17qb_3layer_qnn', '5qb_sqnn', 'uneven_sqnn']
    task1 = 'small_3gates_16bits'
    task2 = 'large_xyz_y_4bits'
    task3 = 'diffsize_xyz_y_844'

    # excel_files = [os.path.join(save_path_small, task, '{}.xls'.format(task)) for task in tasks]
    path1 = os.path.join(save_path_small, task1, '{}.xls'.format(task1))
    path2 = os.path.join(save_path_large, task2, '{}.xls'.format(task2))
    path3 = os.path.join(save_path_large, task3, '{}.xls'.format(task3))

    excel_files = [path1, path2, path3]

    losses = []
    for file in excel_files:
        wb = xlrd.open_workbook(file)
        sheet = wb.sheet_by_index(0)
        values = np.array(sheet.col_values(3)[1:])
        intval = len(values) / 100
        index = np.arange(0, len(values), step=int(intval))
        values = values[index]

        losses.append(values)

    losses = extend(losses)

    f1 = plt.figure()
    x = np.arange(0, len(losses[0]))

    plt.xlabel('epochs')
    plt.ylabel('Training loss')

    for i in range(len(losses)):
        power_smooth = losses[i]

        plt.plot(x, power_smooth, lw=1.5, label=names[i])
    
    plt.legend(loc = 'upper right')
    plt.savefig(figurepath + '/comp_small16_large4_uneven_loss.pdf', dpi=1200)










    



if __name__ == "__main__":
    
    # save_path = './scale_qml/save_small/' 
    # tasks = ["small_6gates_4bits", 'small_6gates_9bits', 'small_6gates_16bits']
    # names = ['5qb_6layer_qnn', '10qb_6layer_qnn', '17qb_6layer_qnn']

    # save_path = './scale_qml/save_large/' 
    # tasks = ["large_xyz_y_4bits", 'large_xyz_y_9bits', 'large_xyz_y_16bits']
    # names = ['5qb_sqnn', '10qb_sqnn', '17qb_sqnn']


    # maxacc = [(4, 81.18), (19, 88.3740884), (31, 92.042042)]
    # maxacc = [(13, 82.0032573), (26, 89.4465894), (57, 90.99)]

    # # large
    # maxacc = [(10, 92.5925926), (72, 93.9809049), (59, 97.467829)]

    # plot_trainloss(save_path, tasks, names)
    # # # plot_trainacc(save_path, tasks, names)
    # plot_valacc(save_path, tasks, names, maxacc)
    # # # plot_testacc(save_path, tasks, names)

    # compare_gates()
    compare_loss()
    
    print('===')

