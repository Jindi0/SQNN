# Scalable Quantum NN
This repo is the code used to produce the results presented in "[Scalable Quantum Neural Network](https://arxiv.org/pdf/2208.07719.pdf)" (SQNN). 

## Content
- `main_qnn.py` builds and trains a regular QNN that follows the circuit structure design in [TensorFlow Quantum Tutorials](https://www.tensorflow.org/quantum/tutorials/mnist#14_encode_the_data_as_quantum_circuits), but uses different encoding method (angle encoding).
- `main_sqnn.py` builds and trains a SQNN that consists of four identical-sized quantum feature extractors and a quantum predictor. The method of data partitioning is illustrated in Fig.5(1st panel) of our paper.
- `main_differsize.py` builds and trains a SQNN that consists of three different-sized quantum feature extractors and a quantum predictor. The method of data partitioning is illustrated in Fig.5(3rd panel) of our paper.
- `data_helper.py` includes the functions for data processing.
- `util.py` contains the functions for recording intermediate results and quantum circuits.

## How to use
### Install requitements
```
    pip install -r requirements.txt
```
### Set hyperparameters
To run the code for the specific tasks, please set the hyperparameters by using the command line or editing the default values of the parameters. The hyperparameters are:

|Hyperparameter|Description|
| :----- | :---- |
|`task`|The name of the current task. A folder with the same name will be created to store intermediate results.|
|`dataset`|The name of dataset|
|`seed`|The random seed used to initialize parameters of VQCs |
|`inputsize`|The downscaled size (nxn) of input data of QNN. If the model is SQNN, the size is that before data splitting. |
|`clfinputsize`|For SQNN, the input size of qunatum predictor.|
|`pieces`|For SQNN, the number of pieces a training instance is splitted into.|
|`lr`|The learning rate|
|`epoch`|The number of training epoch|
|`batchsize`|The size of mini-batch|
|`validation_ratio`|The ratio of training data that is reserved for validation. |

### Run scripts
After setting the hyperparameters, run the scripts by using python3. 
```
    python3 scale_qml/main_qnn.py 
    python3 scale_qml/main.sqnn.py
    python3 scale_qml/main_differsize.py
```

### Check results
A folder with the same name as the task will be created after running the script to store results, and a `.xls` file with the same name as the task will be generated in the folder.

In the `.xls` file, the 1st column stores the hyperparameters of the current task; the 3rd and 4th column record the accuracy and training loss of each mini-batch; the 6th and 7th column record the accuracy and loss on the validation dataset after each training epoch; 9th and 10th column stores the accuracy and loss on the test dataset after each training epoch.

## Contact
If there is any question, please send emails to [jwu21@wm.edu](jwu21@wm.edu).

## Citation
If you use this code in your work, please cite our paper.
