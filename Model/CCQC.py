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

# from CS import CS

def E_ob(state, ob_i, n):
    Ob = paddle.to_tensor(pauli_str_to_matrix([[1.0, f'{ob_i}']], n))
    E_Ob = matmul(matmul(state, Ob), transpose(paddle.conj(state), perm=[0, 2, 1]))
    E_Ob_re = reshape(real(E_Ob), [-1, 1])
    return E_Ob_re


class CCQC(paddle_quantum.Operator):
    """
    Create a model training network
    """
    def __init__(self, n, depth, seed_paras=3407):
        super(CCQC, self).__init__()
        self.n = n
        self.depth = depth
        self.circuit = Circuit(n)
        for d in range(1, depth+1):
            if d % 2:
                for i in range(n):
                    self.circuit.rx(qubits_idx=i)
                    self.circuit.rz(qubits_idx=i)
                    self.circuit.rx(qubits_idx=i)
                self.circuit.cp(qubits_idx=[0, n-1])    
                self.circuit.rx(qubits_idx=n-1)
                for i in range(1, n):
                    self.circuit.cp(qubits_idx=[n-i, n-i-1])
                    self.circuit.rx(qubits_idx=n-i-1)   
            else:
                for i in range(n):
                    self.circuit.rx(qubits_idx=i)
                    self.circuit.rz(qubits_idx=i)
                    self.circuit.rx(qubits_idx=i)
                j = 0
                for i in range(n):
                    nj = (j + (n-3))%n
                    self.circuit.cp(qubits_idx=[j, nj])
                    self.circuit.rx(qubits_idx=nj)
                    j = nj
                       
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
    
