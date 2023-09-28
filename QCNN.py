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

class QCNN(paddle_quantum.Operator):
    """
    Create a model training network
    """
    def __init__(self, n):
        super(QCNN, self).__init__()
        self.n = n
        self.circuit = Circuit(n)
        
        # conv_layer1
        for i in range(0, 10, 2):
            # ansatz 9 of conv_layer_block
            self.circuit.u3(qubits_idx=i)
            self.circuit.u3(qubits_idx=i + 1)
            self.circuit.cnot(qubits_idx=[i, i + 1])
            self.circuit.ry(qubits_idx=i)
            self.circuit.rz(qubits_idx=i + 1)
            self.circuit.cnot(qubits_idx=[i + 1, i])
            self.circuit.ry(qubits_idx=i)
            self.circuit.cnot(qubits_idx=[i, i + 1])
            self.circuit.u3(qubits_idx=i)
            self.circuit.u3(qubits_idx=i + 1)
        for i in range(1, 9, 2):
            # ansatz 9 of conv_layer_block
            self.circuit.u3(qubits_idx=i)
            self.circuit.u3(qubits_idx=i + 1)
            self.circuit.cnot(qubits_idx=[i, i + 1])
            self.circuit.ry(qubits_idx=i)
            self.circuit.rz(qubits_idx=i + 1)
            self.circuit.cnot(qubits_idx=[i + 1, i])
            self.circuit.ry(qubits_idx=i)
            self.circuit.cnot(qubits_idx=[i, i + 1])
            self.circuit.u3(qubits_idx=i)
            self.circuit.u3(qubits_idx=i + 1)
        # ansatz 9 of conv_layer_block
        self.circuit.u3(qubits_idx=0)
        self.circuit.u3(qubits_idx=9)
        self.circuit.cnot(qubits_idx=[0, 9])
        self.circuit.ry(qubits_idx=0)
        self.circuit.rz(qubits_idx=9)
        self.circuit.cnot(qubits_idx=[9, 0])
        self.circuit.ry(qubits_idx=0)
        self.circuit.cnot(qubits_idx=[0, 9])
        self.circuit.u3(qubits_idx=0)
        self.circuit.u3(qubits_idx=9)      
        
        # pool_layer1
        for i in range(0, 10, 2):
            # ansatz of pool_layer_block
            self.circuit.crz(qubits_idx=[i + 1, i])
            self.circuit.x(qubits_idx=i + 1)
            self.circuit.crx(qubits_idx=[i + 1, i])
        
        # conv_layer2
        # ansatz 9 of conv_layer_block
        self.circuit.u3(qubits_idx=0)
        self.circuit.u3(qubits_idx=2)
        self.circuit.cnot(qubits_idx=[0, 2])
        self.circuit.ry(qubits_idx=0)
        self.circuit.rz(qubits_idx=2)
        self.circuit.cnot(qubits_idx=[2, 0])
        self.circuit.ry(qubits_idx=0)
        self.circuit.cnot(qubits_idx=[0, 2])
        self.circuit.u3(qubits_idx=0)
        self.circuit.u3(qubits_idx=2)
        # ansatz 9 of conv_layer_block
        self.circuit.u3(qubits_idx=2)
        self.circuit.u3(qubits_idx=4)
        self.circuit.cnot(qubits_idx=[2, 4])
        self.circuit.ry(qubits_idx=2)
        self.circuit.rz(qubits_idx=4)
        self.circuit.cnot(qubits_idx=[4, 2])
        self.circuit.ry(qubits_idx=2)
        self.circuit.cnot(qubits_idx=[2, 4])
        self.circuit.u3(qubits_idx=2)
        self.circuit.u3(qubits_idx=4)
        # ansatz 9 of conv_layer_block
        self.circuit.u3(qubits_idx=4)
        self.circuit.u3(qubits_idx=6)
        self.circuit.cnot(qubits_idx=[4, 6])
        self.circuit.ry(qubits_idx=4)
        self.circuit.rz(qubits_idx=6)
        self.circuit.cnot(qubits_idx=[6, 4])
        self.circuit.ry(qubits_idx=4)
        self.circuit.cnot(qubits_idx=[4, 6])
        self.circuit.u3(qubits_idx=4)
        self.circuit.u3(qubits_idx=6)
        # ansatz 9 of conv_layer_block
        self.circuit.u3(qubits_idx=6)
        self.circuit.u3(qubits_idx=8)
        self.circuit.cnot(qubits_idx=[6, 8])
        self.circuit.ry(qubits_idx=6)
        self.circuit.rz(qubits_idx=8)
        self.circuit.cnot(qubits_idx=[8, 6])
        self.circuit.ry(qubits_idx=6)
        self.circuit.cnot(qubits_idx=[6, 8])
        self.circuit.u3(qubits_idx=6)
        self.circuit.u3(qubits_idx=8)
        
        # pool_layer2
        # ansatz of pool_layer_block
        self.circuit.crz(qubits_idx=[2, 0])
        self.circuit.x(qubits_idx=2)
        self.circuit.crx(qubits_idx=[2, 0])
        # ansatz of pool_layer_block
        self.circuit.crz(qubits_idx=[6, 4])
        self.circuit.x(qubits_idx=6)
        self.circuit.crx(qubits_idx=[6, 4])         
        
        # ansatz of full connection layer
        self.circuit.cnot(qubits_idx=[0, 4])
        self.circuit.cnot(qubits_idx=[4, 8])
        self.circuit.cnot(qubits_idx=[8, 0])
        self.circuit.rx(qubits_idx=0)
        self.circuit.rx(qubits_idx=4)
        self.circuit.rx(qubits_idx=8)
                       
    def forward(self, state_in, label, classf_n):
        """
        Input：state_in：input quantum state，shape: [BATCH, 1, 2^n]
               label：corresponding to the label，shape: [-1, 1]
        Output:
            Return certain required network information, i.e. acc, loss
        Loss function:
            cross_entropy
        """
        if self.n < (classf_n - 1) * 4:
            raise Exception("num of qubits is less the num of obs, should change the pos of obs, try take z0 x0 on one i")
        state_in.stop_gradient = False
        Utheta = self.circuit.unitary_matrix()
        state_out = matmul(state_in, Utheta)
        E_ob_res = []
        for i in range(classf_n):
            E_ob_res.append(E_ob(state_out, f'z{i*4}', self.n))
        obs = concat(E_ob_res, axis=-1)
        outputs = F.softmax(obs)
        loss = F.cross_entropy(outputs, label)
        acc = mean(cast(argmax(outputs, axis=-1) == label, "float32"))
        return loss, acc, self.circuit, state_in, state_out, outputs
    
def QCNN_MNIST(N, class_nums):
    """
    Input:
        N: qubits num, class_nums: choose classf_mode
    Output:
        Return the Qnn
    """
    if class_nums == 2:
        classf_mode = [3,6]
    else:
        classf_mode = list(range(0,class_nums))
      
    net = QCNN(n=N)
    net_state_dict = paddle.load(f"QCNN_MNIST{classf_mode}clas_{N}qub.pdparams")
    net.set_state_dict(net_state_dict)
      
    return net

def QCNN_Fashion(N, class_nums):
    """
    Input:
        N: qubits num, class_nums: choose classf_mode
    Output:
        Return the Qnn
    """
    if class_nums == 2:
        classf_mode = [1,4]
    else:
        classf_mode = list(range(0,class_nums))
      
    net = QCNN(n=N)
    net_state_dict = paddle.load(f"QCNN_Fashion{classf_mode}clas_{N}qub.pdparams")
    net.set_state_dict(net_state_dict)
      
    return net


if __name__ == '__main__':
    qubit_num = int(sys.argv[1])
    clfn = int(sys.argv[2])
    qcnn_model_softmax(N=qubit_num, class_nums=clfn)
