"""
QuanTest demo
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
from scipy.linalg import sqrtm 
from alive_progress import alive_bar # live progress bar
from Model.QCL import QCL
from Model.CCQC import CCQC
from Model.QCNN import QCNN
from Model.model_load import model_load

paddle.set_device("cpu") # use cpu

model = 'QCL'
dataset = 'MNIST'
class_ns = 2

if class_ns == 2:
    if dataset == 'MNIST':
        classf_mode = [3,6]
    elif dataset == 'Fashion':
        classf_mode = [1,4]
else:
    classf_mode = list(range(0, class_ns))

quantum_test_x = paddle.load(f'testData/q{dataset}test8qn{classf_mode}clf.pqtss')
test_y = np.load(f"testData/{dataset}test_y8qn{classf_mode}clf.npy")
test_y = paddle.to_tensor(test_y, dtype="int64")

test_img_num = 900 * class_ns
# test_img_num = 100
learn_rate = 0.05
iternums = 5
qubit_num = 8
anti_predict_weight = 1 
alpha_FGSM = 1 # 0.1,0.2,0.3,0.4
cov_weight = 1
ent_k = 1
dirname = f'{model}_{dataset}_{classf_mode}' # subdir to store adversial images
if os.path.exists(dirname) == 0:
    os.makedirs(dirname) 
    
adversial_num = 0
iter_list = []
adv_list_x = []
adv_list_y = []
f_list = []
f_sum = 0
t_list = []
t_sum = 0
ent_in_list = []
ent_out_list = []
cov_list = []
cov_sum = 0
total_time = 0
baddatan = 0

net = model_load(N=qubit_num, model=model, dataset=dataset, DEPTH=5, class_nums=class_ns)

time.sleep(1200)
print("Parameters and info:")
print("test_img_num: ", test_img_num, "learn_rate: ", learn_rate, "anti_predict_weight: ", anti_predict_weight, "cov_weight: ", cov_weight, "ent_k: ", ent_k)
print("Output:")
# for per image to test
with alive_bar(test_img_num) as bar:
    for i in range(test_img_num):
        starttime = time.perf_counter()
        print("test img %d:" % i)
        start_time = time.perf_counter()
        ori_img = reshape(quantum_test_x[i], [-1, 1, 2 ** qubit_num])
        ori_label = test_y[i]

        ori_qimg_loss, ori_qimg_acc, ori_qimg_circuit, ori_qimg_state_in, ori_qimg_state_out, ori_qimg_outputs = net(ori_img, ori_label, classf_n=class_ns)
        if ori_qimg_acc == 0:
            baddatan += 1
            continue
        ori_ent_Qorie, ori_ent_in, ori_ent_out = entQ(ori_qimg_state_in, ori_qimg_state_out, ent_k)
        ori_ent_Qnn = ori_ent_out - ori_ent_in
        print("ori_qimg_label:", ori_label.numpy())
        print("ori_qimg_acc:", ori_qimg_acc.numpy())
        print("ori_qimg_outputs: ", ori_qimg_outputs)
        print("ori_ent_out: %f, ori_ent_in: %f, ori_ent_Qnn: %f" % (ori_ent_out, ori_ent_in, ori_ent_Qnn))
        
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
            
            if class_ns == 2:
                obj_orie = DLFuzz2(now_qimg_outputs, ori_qimg_outputs, anti_predict_weight) # orie_deci2
            else:
                obj_orie = DLFuzz3(now_qimg_outputs, ori_qimg_outputs, anti_predict_weight) # orie_deci3
                            
            # obj_orie = FGSM(now_qimg_loss, alpha_FGSM) # orie_cost
            # obj_orie = BIM(now_qimg_loss, alpha_FGSM) # orie_cost

            layer_output = []
            layer_output.append(obj_orie)
            layer_output.append(cov_weight * ent_Qorie)
            grad = paddle.grad(layer_output, now_qimg_state_in)[0]
            perturb = paddle.real(grad).multiply(paddle.to_tensor(learn_rate) )

            ## add corherence_noise_channel
            # now_qimg = corherence_noise_channel(now_qimg, sigma=0.02)
            
            now_qimg = paddle.complex( paddle.clip((paddle.real(now_qimg + perturb)), 0, 1),paddle.to_tensor(0, dtype=paddle.float32))
            now_qimg = now_qimg / paddle.norm(paddle.abs(now_qimg))
            now_qimg_loss, now_qimg_acc, now_qimg_circuit, now_qimg_state_in, now_qimg_state_out, now_qimg_outputs = net(now_qimg, now_label, classf_n=class_ns) 
            print("now_qimg_outputs: ", now_qimg_outputs.numpy())
            ent_Qorie, ent_in, ent_out = entQ(now_qimg_state_in, now_qimg_state_out, ent_k)
            incr_entQnn = ent_out - ent_in - ori_ent_Qnn
            print("out: %f, in: %f, incr_entQnn: %f" % (ent_out, ent_in, incr_entQnn))
            
            
            if now_qimg_acc != 1.0:
                
                print("\t get a gen_adversial_img!!!")
                f = state_fidelity(now_qimg, ori_img)
                t = trace_distance(now_qimg, ori_img)
                print("f: ", f.item())
                print("t: ", t.item())
                adv_qimg = now_qimg
                adversial_num += 1
                f_list.append(f)
                f_sum += f
                t_list.append(t)
                t_sum += t
                cov_sum += ent_out - ent_in
                ent_in_list.append(ent_in)
                ent_out_list.append(ent_out)
                cov_list.append(ent_out - ent_in)
                iter_list.append(iters)
                adv_list_x.append(now_qimg)
                adv_list_y.append(ori_label)
                break
                        
        endtime = time.perf_counter()
        perimgtime = endtime - starttime
        print("perimgtime for img: ", i, " with ", perimgtime)
        total_time = total_time + perimgtime
        bar()
        
QECov = cov_sum / adversial_num
AFM = f_sum / adversial_num
ATD = t_sum / adversial_num
Gen_Rate = adversial_num / (test_img_num - baddatan)

 
print("total_time: " ,total_time, " with " ,test_img_num - baddatan, " images.")
print("adversial_num: " ,adversial_num, " with " ,test_img_num - baddatan, " images.")
print("Metrics:")
print("Gen Rate: ", Gen_Rate, "QECov: ", QECov, "AFM: ", AFM, "ATD: ", ATD)
       

np.save(f"{dirname}/f_list_{model}_{dataset}.npy", f_list)
np.save(f"{dirname}/t_list_{model}_{dataset}.npy", t_list)
np.save(f"{dirname}/ent_in_list_{model}_{dataset}.npy", ent_in_list)
np.save(f"{dirname}/ent_out_list_{model}_{dataset}.npy", ent_out_list)
np.save(f"{dirname}/cov_list_{model}_{dataset}.npy", cov_list)
paddle.save(adv_list_x, f'{dirname}/x_adv_list_{model}_{dataset}.pqtss')
np.save(f"{dirname}/y_adv_list_{model}_{dataset}.npy", adv_list_y)
np.save(f"{dirname}/iter_list_{model}_{dataset}.npy", iter_list)
