"""
QuanTest demo
2023/09/28 
ZiMeng Xiao
CSU, Chang Sha, China
"""
import numpy as np
from matplotlib import pyplot as plt
from paddle import  reshape, argmax, argsort, real, sqrt, divide, matmul, transpose, cast, mean, concat
from paddle.linalg import pinv
from paddle_quantum.qinfo import state_fidelity, trace_distance
import sys
import os
import time
from utils import *
import imageio
imsave = imageio.imsave
from paddle_quantum.ansatz import Circuit
from QCL import QCL_MNIST
from scipy.linalg import sqrtm 
from alive_progress import alive_bar # live progress bar

# paddle.set_device("cpu") 
quantum_test_x = paddle.load('Data/test_xMNIST[3, 6].pqtss')
test_y = np.load("Data/test_yMNIST[3, 6].npy")
test_y = paddle.to_tensor(test_y, dtype="int64")

adversial_num = 0
test_img_num = 10
learn_rate = 0.05
# iternums = 5
qubit_num = 10
anti_predict_weight = 1 
alpha_FGSM = 1
cov_weight = 1
ent_k = 1
f_sum = 0
t_sum = 0
cov_sum = 0
total_time = 0
baddatan = 0

QCL_MNIST_net = QCL_MNIST(N=qubit_num, DEPTH=5, class_nums=2)

print("Parameters and info:")
print("test_img_num: ", test_img_num, "learn_rate: ", learn_rate,  "anti_predict_weight: ", anti_predict_weight, "cov_weight: ", cov_weight, "ent_k: ", ent_k)
# for per image to test
with alive_bar(test_img_num) as bar:
    for i in range(test_img_num):
        starttime = time.perf_counter()
        print("test img %d:" % i)
        start_time = time.perf_counter()
        ori_img = reshape(quantum_test_x[i], [-1, 1, 2 ** qubit_num])
        ori_label = test_y[i]
        bar()
        ori_qimg_loss, ori_qimg_acc, ori_qimg_circuit, ori_qimg_state_in, ori_qimg_state_out, ori_qimg_outputs = QCL_MNIST_net(ori_img, ori_label, classf_n=2)
        if ori_qimg_acc == 0:
            baddatan += 1
            continue
        ori_ent_Qorie, ori_ent_in, ori_ent_out = entQ(ori_qimg_state_in, ori_qimg_state_out, ent_k)
        ori_ent_Qnn = ori_ent_out - ori_ent_in
        
        now_qimg_loss, now_qimg_acc, now_qimg_circuit, now_qimg_state_in, now_qimg_state_out, now_qimg_outputs = ori_qimg_loss, ori_qimg_acc, ori_qimg_circuit, ori_qimg_state_in, ori_qimg_state_out, ori_qimg_outputs
        ent_Qorie, now_ent_in, now_ent_out = ori_ent_Qorie, ori_ent_in, ori_ent_out
        now_qimg, now_label = ori_img, ori_label
        
        iters = 0
        while True:
            print("\n>>>iters %d : " % iters)
            iters += 1
            ## may paddlequantum will give bad datax that failed to find a square root, so
            if True in np.isnan(sqrtm(paddle_quantum.intrinsic._type_transform((now_qimg), "density_matrix").numpy())):
                baddatan += 1
                break
            
            obj_orie = DLFuzz2(now_qimg_outputs, ori_qimg_outputs, anti_predict_weight) # orie_deci2
            # obj_orie = DLFuzz3(now_qimg_outputs, ori_qimg_outputs, anti_predict_weight) # orie_deci3
            # obj_orie = FGSM(now_qimg_loss, alpha_FGSM) # orie_cost

            layer_output = []
            layer_output.append(obj_orie)
            layer_output.append(cov_weight * ent_Qorie)
            grad = paddle.grad(layer_output, now_qimg_state_in)[0]
            perturb = paddle.real(grad).multiply(paddle.to_tensor(learn_rate) )

            ## add corherence_noise_channel
            # now_qimg = corherence_noise_channel(now_qimg, sigma=0.02)
            
            now_qimg = paddle.complex( paddle.clip((paddle.real(now_qimg + perturb)), 0, 1),paddle.to_tensor(0, dtype=paddle.float32))
            now_qimg = now_qimg / paddle.norm(paddle.abs(now_qimg))
            now_qimg_loss, now_qimg_acc, now_qimg_circuit, now_qimg_state_in, now_qimg_state_out, now_qimg_outputs = QCL_MNIST_net(now_qimg, now_label, classf_n=2) 
            ent_Qorie, ent_in, ent_out = entQ(now_qimg_state_in, now_qimg_state_out, ent_k)
            incr_entQnn = ent_out - ent_in - ori_ent_Qnn
            f = state_fidelity(now_qimg, ori_img)
            t = trace_distance(now_qimg, ori_img)

            
            if now_qimg_acc != 1.0 and (f > 0.95 or t < 0.3):
                print("\t get a gen_adversial_img!!!")
                adv_qimg = now_qimg
                adversial_num += 1
                f_sum += f
                t_sum += t
                cov_sum += ent_out - ent_in
                break
            if (f <= 0.95 and t >= 0.3):
                print("Faild in this image. :( ")
                break

        endtime = time.perf_counter()
        perimgtime = endtime - starttime
        print("perimgtime for img: ", i, " with ", perimgtime)
        total_time = total_time + perimgtime
        
        
QECov = cov_sum / adversial_num
AFM = f_sum / adversial_num
ATD = t_sum / adversial_num
Gen_Rate = adversial_num / (test_img_num - baddatan)

print("total_time: " ,total_time, " with " ,test_img_num - baddatan, " images.")
print("adversial_num: " ,adversial_num, " with " ,test_img_num - baddatan, " images.")
print("Metrics:")
print("Gen Rate: ", Gen_Rate, "QECov: ", QECov, "AFM: ", AFM, "ATD: ", ATD)