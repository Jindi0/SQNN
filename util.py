import xlwt
from cirq.contrib.svg import circuit_to_svg

def save_text(src: str, dest_path=None, file_ext="svg"):
    if dest_path is None:
        import uuid
        dest_path = str(uuid.uuid4()) + ".{}".format(file_ext)
    with open(dest_path, 'w', encoding='utf-8') as f: 
        f.write(src)

def dump_circuit(circuit: 'cirq.Circuit', dest_path=None):
    save_text(circuit_to_svg(circuit), dest_path=dest_path)



def init_log(args):
    '''
        The log file generator for QNN model
        The file will record the training accuracy/loss in each training iteration,
        the accuracy/loss on the validation/test dataset after each epoch,
        and the hyperparameters.
    '''
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('results',cell_overwrite_ok=True)
    row = ['settings']
    row += '-'
    row.append(['train_acc'])
    row.append(['train_loss'])
    row += '-'
    row.append(['val_acc'])
    row.append(['val_loss'])
    row += '-'
    row.append(['test_acc'])
    row.append(['test_loss'])

    style = xlwt.XFStyle()
    style.alignment.wrap = 1

    for i in range(len(row)):
        sheet1.write(0, i, row[i])

    sheet1.write(1, 0, 'Batch_size = ' + str(args.batchsize))
    sheet1.write(2, 0, 'Learning_rate = ' + str(args.lr))
    sheet1.write(3, 0, 'Input_size = ' + str(args.inputsize))

    return f, sheet1

