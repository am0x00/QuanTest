"""
tools
"""
import sys
import os
import math
import paddle
import numpy as np
from paddle import reshape, argmax, argsort, real, concat, sqrt, divide, matmul
import paddle_quantum
from paddle_quantum.gate import AmplitudeEncoding
from paddle_quantum.dataset import MNIST
from paddle_quantum.linalg import abs_norm, NKron, is_unitary
from datetime import datetime
from paddle_quantum.ansatz import Circuit

def liner_map(b, j, psi):
    """
    Input:
        b, j, psi: elem-parameters from MW-enaglement
    Output:
        Reuturn liner map for MW-enaglement measure
    """
    newpsi = []
    num_qubits = math.ceil(math.log2(psi.size) )
    for i in range(psi.size):
        delta_i2bin = ((i>>(num_qubits-1-j))&1) ^ b ^ 1
        if(delta_i2bin):
            newpsi.append(psi[i])
    return concat(newpsi)


def gener_distance(u, v):
    """
    Input:
        u, v: parameters from MW-enaglement
    Output:
        Returns the general distance between u and v for MW-enaglement measure
    """
    uvmat = NKron(u, v) - NKron(v, u)
    return 1/2 * paddle.norm(paddle.abs(uvmat))**2

def ent_state(psi):
    """
    Input
        psi: target quantum state    
    Output:
        Returns the MW-enaglement measure of psi
    """
    num_qubits = math.ceil(math.log2(psi.size) )
    res = 0.0
    psi_np = psi.numpy()
    for j in range(num_qubits):
        res += gener_distance(liner_map(0, j, psi), liner_map(1, j, psi))
    return res * 4 / num_qubits

def entQ(now_gimg_state_in, now_gimg_state_out, k):
    ent_psi_statein = ent_state(now_gimg_state_in[0,0])
    ent_psi_stateout = ent_state(now_gimg_state_out[0,0])
    return ent_psi_stateout - k * ent_psi_statein, ent_psi_statein, ent_psi_stateout
    

def DLFuzz2(now_outputs, ori_outputs, w):
    """
    Input:
        now_outputs, ori_outputs: now and origin outputs by QNN, w: weight to anti
    Output:
        Return the decision boundrary orientation
    Decision boundary orientration from DLFuzz for 2 classfication
    """
    loss1 = now_outputs[0, argmax(ori_outputs[0])]
    loss2 = now_outputs[0, argsort(ori_outputs[0])[-2]]
    return w * loss2 - loss1

def FGSM(loss, alhpa):
    """
    Input:
        loss, alhpa: parameters form FGSM
    Output:
        Return the cost orientation
    Cost orientation from FGSM
    """
    return alhpa * loss
    
def DLFuzz3(now_outputs, ori_outputs, w):
    """
    Input:
        now_outputs, ori_outputs: now and origin outputs by QNN, w: weight to anti
    Output:
        Return the decision boundrary orientation
    Decision boundary orientration from DLFuzz for 10 classfication
    """
    args_ori = argsort(ori_outputs[0])
    loss1 = now_outputs[0, argmax(ori_outputs[0])]
    loss2 = now_outputs[0, args_ori[-2]]
    loss3 = now_outputs[0, args_ori[-3]]
    return w * (loss2 + loss3) - loss1

def corherence_noise_channel(input_state, sigma=0.01):
    """
    Input:
        input_state, sigma: std
    Output:
        Return the unitary of random perturbation operator
    Add a random perturbation operator on input_state with std=sigma, mean=0
    """
    num_qubits = math.ceil(math.log2(input_state.size) )
    
    Noisecir = Circuit(num_qubits)
    for i in range(num_qubits):
        Noisecir.u3(qubits_idx=i)
        # Noisecir.ry(qubits_idx=i)
        # Noisecir.rz(qubits_idx=i)

    Noisecir.randomize_param(arg0=0, arg1=sigma*math.pi, initializer_type='Normal')
    Utheta = Noisecir.unitary_matrix()
    
    return matmul(input_state, Utheta)
