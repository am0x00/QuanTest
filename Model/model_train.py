import paddle
from paddle import matmul, transpose, reshape, argmax, real, cast, mean, concat
import paddle_quantum
from paddle_quantum.ansatz import Circuit
from paddle_quantum.linalg import abs_norm
from paddle_quantum.gate import AmplitudeEncoding
from paddle_quantum.dataset import MNIST
from paddle_quantum.qinfo import pauli_str_to_matrix
import sys
#sys.path.append("..")
import paddle.nn.functional as F
import numpy as np

from QCL import QCL
from CCQC import CCQC
from QCNN import QCNN

def CCQC_MNIST(N, DEPTH, class_nums):
    """
    Input:
        N: qubits num, DEPTH: deepth of Qnn, class_nums: number of classfication tasks.
    Output:
        Return the trained QNN
    """
    if class_nums == 2:
        classf_mode = [3,6]
    else:
        classf_mode = list(range(0,class_nums))

    testyload = np.load(f"data/MNISTtest_y{N}qn{classf_mode}clf.npy")
    trainyload = np.load(f"data/MNISTtrain_y{N}qn{classf_mode}clf.npy")
    qtestxload = paddle.load(f"data/qMNISTtest{N}qn{classf_mode}clf.pqtss")
    qtrainxload = paddle.load(f"data/qMNISTtrain{N}qn{classf_mode}clf.pqtss")
    LR = 0.01 
    BATCH = 60*class_nums
    trainyload = paddle.to_tensor(trainyload, dtype="int64")
    testyload = paddle.to_tensor(testyload, dtype="int64")
    N_train, in_dim = qtrainxload.shape
    EPOCH = 10
    paddle.seed(1)
    net = CCQC(n=N, depth=DEPTH)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())
    for ep in range(EPOCH):
        for itr in range(N_train // BATCH):
            input_state = qtrainxload[itr * BATCH:(itr + 1) * BATCH]
            input_state = reshape(input_state, [-1, 1, 2 ** N])
            label = trainyload[itr * BATCH:(itr + 1) * BATCH]
            test_input_state = reshape(qtestxload, [-1, 1, 2 ** N])
            train_loss, train_acc, cir, state_in, state_out, outputs = net(state_in=input_state, label=label, classf_n=class_nums)
            if itr % 5 == 0:
                loss_useless, test_acc, t_cir, statein, stateout, outputs = net(state_in=test_input_state, label=testyload, classf_n=class_nums)
                print("epoch:", ep, "iter:", itr,
                      "loss: %.4f" % train_loss.numpy(),
                      "train acc: %.4f" % train_acc,
                      "test acc: %.4f" % test_acc)
            train_loss.backward()
            opt.minimize(train_loss)
            opt.clear_grad()
    paddle.save(net.state_dict(), f"CCQC_MNIST_{classf_mode}clas_{N}qub_{DEPTH}dep.pdparams") 
    return net

def CCQC_Fashion(N, DEPTH, class_nums):
    """
    Input:
        N: qubits num, DEPTH: deepth of Qnn, class_nums: number of classfication tasks.
    Output:
        Return the trained QNN
    """
    if class_nums == 2:
        classf_mode = [1,4]
    else:
        classf_mode = list(range(0,class_nums))

    testyload = np.load(f"data/Fashiontest_y{N}qn{classf_mode}clf.npy")
    trainyload = np.load(f"data/Fashiontrain_y{N}qn{classf_mode}clf.npy")
    qtestxload = paddle.load(f"data/qFashiontest{N}qn{classf_mode}clf.pqtss")
    qtrainxload = paddle.load(f"data/qFashiontrain{N}qn{classf_mode}clf.pqtss")
    LR = 0.01 
    BATCH = 60*class_nums
    trainyload = paddle.to_tensor(trainyload, dtype="int64")
    testyload = paddle.to_tensor(testyload, dtype="int64")
    N_train, in_dim = qtrainxload.shape
    EPOCH = 10
    paddle.seed(1)
    net = CCQC(n=N, depth=DEPTH)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())
    for ep in range(EPOCH):
        for itr in range(N_train // BATCH):
            input_state = qtrainxload[itr * BATCH:(itr + 1) * BATCH]
            input_state = reshape(input_state, [-1, 1, 2 ** N])
            label = trainyload[itr * BATCH:(itr + 1) * BATCH]
            test_input_state = reshape(qtestxload, [-1, 1, 2 ** N])
            train_loss, train_acc, cir, state_in, state_out, outputs = net(state_in=input_state, label=label, classf_n=class_nums)
            if itr % 5 == 0:
                loss_useless, test_acc, t_cir, statein, stateout, outputs = net(state_in=test_input_state, label=testyload, classf_n=class_nums)
                print("epoch:", ep, "iter:", itr,
                      "loss: %.4f" % train_loss.numpy(),
                      "train acc: %.4f" % train_acc,
                      "test acc: %.4f" % test_acc)
            train_loss.backward()
            opt.minimize(train_loss)
            opt.clear_grad()
    paddle.save(net.state_dict(), f"CCQC_Fashion_{classf_mode}clas_{N}qub_{DEPTH}dep.pdparams") 
    return net

def QCL_MNIST(N, DEPTH, class_nums):
    """
    Input:
        N: qubits num, DEPTH: deepth of Qnn, class_nums: number of classfication tasks.
    Output:
        Return the trained QNN
    """
    if class_nums == 2:
        classf_mode = [3,6]
    else:
        classf_mode = list(range(0,class_nums))

    testyload = np.load(f"data/MNISTtest_y{N}qn{classf_mode}clf.npy")
    trainyload = np.load(f"data/MNISTtrain_y{N}qn{classf_mode}clf.npy")
    qtestxload = paddle.load(f"data/qMNISTtest{N}qn{classf_mode}clf.pqtss")
    qtrainxload = paddle.load(f"data/qMNISTtrain{N}qn{classf_mode}clf.pqtss")
    LR = 0.01 
    BATCH = 60*class_nums
    trainyload = paddle.to_tensor(trainyload, dtype="int64")
    testyload = paddle.to_tensor(testyload, dtype="int64")
    N_train, in_dim = qtrainxload.shape
    EPOCH = 10
    paddle.seed(1)
    net = QCL(n=N, depth=DEPTH)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())
    for ep in range(EPOCH):
        for itr in range(N_train // BATCH):
            input_state = qtrainxload[itr * BATCH:(itr + 1) * BATCH]
            input_state = reshape(input_state, [-1, 1, 2 ** N])
            label = trainyload[itr * BATCH:(itr + 1) * BATCH]
            test_input_state = reshape(qtestxload, [-1, 1, 2 ** N])
            train_loss, train_acc, cir, state_in, state_out, outputs = net(state_in=input_state, label=label, classf_n=class_nums)
            if itr % 5 == 0:
                loss_useless, test_acc, t_cir, statein, stateout, outputs = net(state_in=test_input_state, label=testyload, classf_n=class_nums)
                print("epoch:", ep, "iter:", itr,
                      "loss: %.4f" % train_loss.numpy(),
                      "train acc: %.4f" % train_acc,
                      "test acc: %.4f" % test_acc)
            train_loss.backward()
            opt.minimize(train_loss)
            opt.clear_grad()
    paddle.save(net.state_dict(), f"QCL_MNIST_{classf_mode}clas_{N}qub_{DEPTH}dep.pdparams") 
    return net

def QCL_Fashion(N, DEPTH, class_nums):
    """
    Input:
        N: qubits num, DEPTH: deepth of Qnn, class_nums: number of classfication tasks.
    Output:
        Return the trained QNN
    """
    if class_nums == 2:
        classf_mode = [1,4]
    else:
        classf_mode = list(range(0,class_nums))

    testyload = np.load(f"data/Fashiontest_y{N}qn{classf_mode}clf.npy")
    trainyload = np.load(f"data/Fashiontrain_y{N}qn{classf_mode}clf.npy")
    qtestxload = paddle.load(f"data/qFashiontest{N}qn{classf_mode}clf.pqtss")
    qtrainxload = paddle.load(f"data/qFashiontrain{N}qn{classf_mode}clf.pqtss")
    LR = 0.01 
    BATCH = 60*class_nums
    trainyload = paddle.to_tensor(trainyload, dtype="int64")
    testyload = paddle.to_tensor(testyload, dtype="int64")
    N_train, in_dim = qtrainxload.shape
    EPOCH = 10
    paddle.seed(1)
    net = QCL(n=N, depth=DEPTH)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())
    for ep in range(EPOCH):
        for itr in range(N_train // BATCH):
            input_state = qtrainxload[itr * BATCH:(itr + 1) * BATCH]
            input_state = reshape(input_state, [-1, 1, 2 ** N])
            label = trainyload[itr * BATCH:(itr + 1) * BATCH]
            test_input_state = reshape(qtestxload, [-1, 1, 2 ** N])
            train_loss, train_acc, cir, state_in, state_out, outputs = net(state_in=input_state, label=label, classf_n=class_nums)
            if itr % 5 == 0:
                loss_useless, test_acc, t_cir, statein, stateout, outputs = net(state_in=test_input_state, label=testyload, classf_n=class_nums)
                print("epoch:", ep, "iter:", itr,
                      "loss: %.4f" % train_loss.numpy(),
                      "train acc: %.4f" % train_acc,
                      "test acc: %.4f" % test_acc)
            train_loss.backward()
            opt.minimize(train_loss)
            opt.clear_grad()
    paddle.save(net.state_dict(), f"QCL_Fashion_{classf_mode}clas_{N}qub_{DEPTH}dep.pdparams") 
    return net

def QCNN_MNIST(N, class_nums):
    """
    Input:
        N: qubits num,  class_nums: number of classfication tasks.
    Output:
        Return the trained QNN
    """
    if class_nums == 2:
        classf_mode = [3,6]
    else:
        classf_mode = list(range(0,class_nums))

    testyload = np.load(f"data/MNISTtest_y{N}qn{classf_mode}clf.npy")
    trainyload = np.load(f"data/MNISTtrain_y{N}qn{classf_mode}clf.npy")
    qtestxload = paddle.load(f"data/qMNISTtest{N}qn{classf_mode}clf.pqtss")
    qtrainxload = paddle.load(f"data/qMNISTtrain{N}qn{classf_mode}clf.pqtss")
    LR = 0.01 
    BATCH = 60*class_nums
    trainyload = paddle.to_tensor(trainyload, dtype="int64")
    testyload = paddle.to_tensor(testyload, dtype="int64")
    N_train, in_dim = qtrainxload.shape
    EPOCH = 10
    paddle.seed(1)
    net = QCNN(n=8)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())
    for ep in range(EPOCH):
        for itr in range(N_train // BATCH):
            input_state = qtrainxload[itr * BATCH:(itr + 1) * BATCH]
            input_state = reshape(input_state, [-1, 1, 2 ** N])
            label = trainyload[itr * BATCH:(itr + 1) * BATCH]
            test_input_state = reshape(qtestxload, [-1, 1, 2 ** N])
            train_loss, train_acc, cir, state_in, state_out, outputs = net(state_in=input_state, label=label, classf_n=class_nums)
            if itr % 5 == 0:
                loss_useless, test_acc, t_cir, statein, stateout, outputs = net(state_in=test_input_state, label=testyload, classf_n=class_nums)
                print("epoch:", ep, "iter:", itr,
                      "loss: %.4f" % train_loss.numpy(),
                      "train acc: %.4f" % train_acc,
                      "test acc: %.4f" % test_acc)
            train_loss.backward()
            opt.minimize(train_loss)
            opt.clear_grad()
    paddle.save(net.state_dict(), f"QCNN_MNIST_{classf_mode}clas_{N}qub.pdparams") 
    return net

def QCNN_Fashion(N, class_nums):
    """
    Input:
        N: qubits num,  class_nums: number of classfication tasks.
    Output:
        Return the trained QNN
    """
    if class_nums == 2:
        classf_mode = [1,4]
    else:
        classf_mode = list(range(0,class_nums))

    testyload = np.load(f"data/Fashiontest_y{N}qn{classf_mode}clf.npy")
    trainyload = np.load(f"data/Fashiontrain_y{N}qn{classf_mode}clf.npy")
    qtestxload = paddle.load(f"data/qFashiontest{N}qn{classf_mode}clf.pqtss")
    qtrainxload = paddle.load(f"data/qFashiontrain{N}qn{classf_mode}clf.pqtss")
    LR = 0.01 
    BATCH = 60*class_nums
    trainyload = paddle.to_tensor(trainyload, dtype="int64")
    testyload = paddle.to_tensor(testyload, dtype="int64")
    N_train, in_dim = qtrainxload.shape
    EPOCH = 10
    paddle.seed(1)
    net = QCNN(n=8)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())
    for ep in range(EPOCH):
        for itr in range(N_train // BATCH):
            input_state = qtrainxload[itr * BATCH:(itr + 1) * BATCH]
            input_state = reshape(input_state, [-1, 1, 2 ** N])
            label = trainyload[itr * BATCH:(itr + 1) * BATCH]
            test_input_state = reshape(qtestxload, [-1, 1, 2 ** N])
            train_loss, train_acc, cir, state_in, state_out, outputs = net(state_in=input_state, label=label, classf_n=class_nums)
            if itr % 5 == 0:
                loss_useless, test_acc, t_cir, statein, stateout, outputs = net(state_in=test_input_state, label=testyload, classf_n=class_nums)
                print("epoch:", ep, "iter:", itr,
                      "loss: %.4f" % train_loss.numpy(),
                      "train acc: %.4f" % train_acc,
                      "test acc: %.4f" % test_acc)
            train_loss.backward()
            opt.minimize(train_loss)
            opt.clear_grad()
    paddle.save(net.state_dict(), f"QCNN_Fashion_{classf_mode}clas_{N}qub.pdparams") 
    return net

if __name__ == '__main__':
    qubit_num = int(sys.argv[1])
    qnn_depth = int(sys.argv[2])
    clfn = int(sys.argv[3])
    model = str(sys.argv[4])
    dataset = str(sys.argv[5])
    if model == 'CCQC':
        if dataset == 'MNIST':
            CCQC_MNIST(N=qubit_num, DEPTH=qnn_depth, class_nums=clfn)
        elif dataset == 'Fashion':
            CCQC_Fashion(N=qubit_num, DEPTH=qnn_depth, class_nums=clfn)
        else:
            raise Exception("Dataset does not exist.")
    elif model == 'QCL':
        if dataset == 'MNIST':
            QCL_MNIST(N=qubit_num, DEPTH=qnn_depth, class_nums=clfn)
        elif dataset == 'Fashion':
            QCL_Fashion(N=qubit_num, DEPTH=qnn_depth, class_nums=clfn)
        else:
            raise Exception("Dataset does not exist.")    
    elif model == 'QCNN':
        if dataset == 'MNIST':
            QCNN_MNIST(N=qubit_num, class_nums=clfn)
        elif dataset == 'Fashion':
            QCNN_Fashion(N=qubit_num, class_nums=clfn)
        else:
            raise Exception("Dataset does not exist.")
    else:
        raise Exception("Model does not exist.")