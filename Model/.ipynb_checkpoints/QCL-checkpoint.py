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

def E_ob(state, ob_i, n):
    Ob = paddle.to_tensor(pauli_str_to_matrix([[1.0, f'{ob_i}']], n))
    E_Ob = matmul(matmul(state, Ob), transpose(paddle.conj(state), perm=[0, 2, 1]))
    E_Ob_re = reshape(real(E_Ob), [-1, 1])
    return E_Ob_re


class QCL(paddle_quantum.Operator):
    """
    Create a model training network
    """
    def __init__(self, n, depth):
        super(QCL, self).__init__()
        self.n = n
        self.depth = depth
        self.circuit = Circuit(n)
        for d in range(depth):
            for i in range(n-1):
                self.circuit.cnot(qubits_idx=[i, i + 1])         
            self.circuit.cnot(qubits_idx=[n-1, 0])
            for i in range(n):
                self.circuit.rx(qubits_idx=i)
                self.circuit.rz(qubits_idx=i)
                self.circuit.rx(qubits_idx=i)
                       
    def forward(self, state_in, label, classf_n):
        """
        Input：state_in：input quantum state，shape: [BATCH, 1, 2^n]
               label：corresponding to the label，shape: [-1, 1]
        Output:
            Return certain required network information, i.e. acc, loss
        Loss function:
            cross_entropy
        """
        if self.n < classf_n:
            raise Exception("num of qubits is less the num of obs, should change the pos of obs, try take z0 x0 on one i")
        state_in.stop_gradient = False
        Utheta = self.circuit.unitary_matrix()
        state_out = matmul(state_in, Utheta)
        E_ob_res = []
        for i in range(classf_n):
            E_ob_res.append(E_ob(state_out, f'z{i}', self.n))
        obs = concat(E_ob_res, axis=-1)
        outputs = F.softmax(obs)
        loss = F.cross_entropy(outputs, label)
        acc = mean(cast(argmax(outputs, axis=-1) == label, "float32"))
        return loss, acc, self.circuit, state_in, state_out, outputs
    
def qcl_model_softmax(N, train, DEPTH, class_nums):
    """
    Input:
        N: qubits num, train: train or read parms, DEPTH: deepth of Qnn
    Output:
        Return the Qnn for 10 classf
    """
    if class_nums == 2:
        classf_mode = [1,4]
    else:
        classf_mode = list(range(0,class_nums))
    if train:

        qtrainxload = paddle.load(f'qFashionMNISTtrain{N}qn{classf_mode}clfFULL.pqtss')
        qtestxload = paddle.load(f'qFashionMNISTtest{N}qn{classf_mode}clfFULL.pqtss')
        trainyload = np.load(f"train_y{N}qn{classf_mode}clfFULL.npy")
        testyload = np.load(f"test_y{N}qn{classf_mode}clfFULL.npy")
        LR = 0.01   
        BATCH = 70*class_nums
        trainyload = paddle.to_tensor(trainyload, dtype="int64")
        testyload = paddle.to_tensor(testyload, dtype="int64")
        N_train, in_dim = qtrainxload.shape
        EPOCH = int(200 * BATCH / N_train)
        paddle.seed(1)
        
    net = QCL(n=N, depth=DEPTH)
    
    if train:
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
                
        paddle.save(net.state_dict(), f"sm_qcl_model{classf_mode}clas_{N}qub_{DEPTH}dep.pdparams") 
        
    else:
        net_state_dict = paddle.load(f"sm_qcl_model{classf_mode}clas_{N}qub_{DEPTH}dep.pdparams")
        net.set_state_dict(net_state_dict)
        
    return net

# def qcl_model2(N, train, DEPTH):
#     """
#     Input:
#         N: qubits num, train: train or read parms, DEPTH: deepth of Qnn
#     Output:
#         Return the Qnn for 2 classf
#     """
#     if train:
#         (qtrainxload, trainyload), (qtestxload, testyload) = load_data_MNIST(Q=True, code_qubit_num=N, classes=[3,6], train_n = 1400, test_n = 300)
#         print("data load finish.")
#         EPOCH = 5
#         LR = 0.05   
#         BATCH = 100
#         trainyload = paddle.to_tensor(trainyload, dtype="int64")
#         testyload = paddle.to_tensor(testyload, dtype="int64")
#         N_train, in_dim = qtrainxload.shape
#         paddle.seed(1)
        
#     net = QCL(n=N, depth=DEPTH)
    
#     if train:
#         opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())
#         for ep in range(EPOCH):
#             for itr in range(N_train // BATCH):
#                 input_state = qtrainxload[itr * BATCH:(itr + 1) * BATCH]  
#                 input_state = reshape(input_state, [-1, 1, 2 ** N])
#                 label = trainyload[itr * BATCH:(itr + 1) * BATCH]
#                 test_input_state = reshape(qtestxload, [-1, 1, 2 ** N])
#                 train_loss, train_acc, cir, state_in, state_out, outputs = net(state_in=input_state, label=label, classf_n=2)
#                 if itr % 5 == 0:
#                     loss_useless, test_acc, t_cir, statein, stateout, outputs = net(state_in=test_input_state, label=testyload, classf_n=2)
#                     print("epoch:", ep, "iter:", itr,
#                           "loss: %.4f" % train_loss.numpy(),
#                           "train acc: %.4f" % train_acc,
#                           "test acc: %.4f" % test_acc)
#                 train_loss.backward()
#                 opt.minimize(train_loss)
#                 opt.clear_grad()
                
#         paddle.save(net.state_dict(), f"qcl_model2clas_{N}qub_{DEPTH}dep.pdparams") 
        
#     net_state_dict = paddle.load(f"qcl_model2clas_{N}qub_{DEPTH}dep.pdparams")
#     net.set_state_dict(net_state_dict)
        
#     return net

if __name__ == '__main__':
    qubit_num = int(sys.argv[1])
    qnn_depth = int(sys.argv[2])
    clfn = int(sys.argv[3])
    qcl_model_softmax(N=qubit_num, train=True, DEPTH=qnn_depth, class_nums=clfn)
    # if classf_mode == 2:
    #     qcl_model2(N=qubit_num, train=True, DEPTH=qnn_depth)
    # elif classf_mode == 10: # wait for 3?
    #     qcl_model10(N=qubit_num, train=True, DEPTH=qnn_depth)
    # else:
    #     raise Exception("classf_mode only can be 2 or 10.")