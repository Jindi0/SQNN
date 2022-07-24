import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import collections
from sklearn.utils import shuffle


def filter_36(x, y):
    keep = (y == 3) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 3
    return x,y


def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       orig_x[tuple(x.flatten())] = x
       mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Throw out images that match more than one label.
          pass

    return np.array(new_x), np.array(new_y)


def load_raw_data(args):
    '''
        Load training data from MNIST dataset
        For the binary classification task, only 3 and 6 are kept for training
    '''

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,2pi] range.
    x_train, x_test = x_train[..., np.newaxis]*2*np.pi/255.0, x_test[..., np.newaxis]*2*np.pi/255.0

    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))

    x_train, y_train = filter_36(x_train, y_train)
    x_test, y_test = filter_36(x_test, y_test)

    print("Number of filtered training examples:", len(x_train))
    print("Number of filtered test examples:", len(x_test))

    # Downscale the images
    x_train_small = tf.image.resize(x_train, (args.inputsize, args.inputsize)).numpy()
    x_test_small = tf.image.resize(x_test, (args.inputsize, args.inputsize)).numpy()

    # Remove contradictory examples
    x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
    x_test_small, y_test = remove_contradicting(x_test_small, y_test)
    x_train_nocon, y_train_nocon = shuffle(x_train_nocon, y_train_nocon)

    return x_train_nocon, y_train_nocon, x_test_small, y_test



def split_train_validation(x, y, ratio):
    x, y = shuffle(x, y)
    number = int(len(x) * ratio)

    return x[number:], y[number:], x[:number], y[:number]



def shuffle_dataset(x, y):
    x0, x1, x2, x3, y = shuffle(x[0], x[1], x[2], x[3], y)
    return [x0, x1, x2, x3], y



def img_split(args, x, y):
    '''
        Split each training image into 4 segments with the same size,
        as shown the first panel of figure 5 in our paper.
    '''
    x_piece1 = []
    x_piece2 = []
    x_piece3 = []
    x_piece4 = []
    y_split = []
    inputsize = int(args.inputsize / 2)
    for i in range(len(x)):
        img = x[i]
        x_piece1.append(img[0:inputsize, 0:inputsize].flatten())
        x_piece2.append(img[0:inputsize, inputsize:].flatten())
        x_piece3.append(img[inputsize:, 0:inputsize].flatten())
        x_piece4.append(img[inputsize:, inputsize:].flatten())
        y_split.append(y[i])

    return [x_piece1, x_piece2, x_piece3, x_piece4], np.array(y_split)

def img_split_3piece(args, x, y):
    '''
        Split each training image into 3 segments with the different sizes,
        as shown the third panel of figure 5 in our paper.
    '''
    x_piece1 = []
    x_piece2 = []
    x_piece3 = []
    
    y_split = []
    inputsize = int(args.inputsize / 2)
    for i in range(len(x)):
        img = x[i]
        x_piece1.append(img[0:inputsize, :].flatten())
        x_piece2.append(img[inputsize:, 0:inputsize:].flatten())
        x_piece3.append(img[inputsize:, inputsize:].flatten())
        y_split.append(y[i])

    return [x_piece1, x_piece2, x_piece3], np.array(y_split)


