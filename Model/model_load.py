import paddle
from Model.QCL import QCL
from Model.CCQC import CCQC
from Model.QCNN import QCNN

def model_load(N, model, dataset, DEPTH, class_nums):
    if dataset == 'MNIST':
        if class_nums == 2:
            classf_mode = [3,6]
        else:
            classf_mode = list(range(0,class_nums))
        if model == 'CCQC':
            net = CCQC(n=N, depth=DEPTH)
            net_state_dict = paddle.load(f"Model/CCQC_MNIST_{classf_mode}clas_{N}qub_{DEPTH}dep.pdparams")
            net.set_state_dict(net_state_dict)
        elif model == 'QCL':
            net = QCL(n=N, depth=DEPTH)
            net_state_dict = paddle.load(f"Model/QCL_MNIST_{classf_mode}clas_{N}qub_{DEPTH}dep.pdparams")
            net.set_state_dict(net_state_dict)
        elif model == 'QCNN':
            net = QCNN(n=N)
            net_state_dict = paddle.load(f"Model/QCNN_MNIST_{classf_mode}clas_{N}qub.pdparams")
            net.set_state_dict(net_state_dict)
        else:
            raise Exception("Model does not exist.")
    elif dataset == 'Fashion':
        if class_nums == 2:
            classf_mode = [1,4]
        else:
            classf_mode = list(range(0,class_nums))
        if model == 'CCQC':
            net = CCQC(n=N, depth=DEPTH)
            net_state_dict = paddle.load(f"Model/CCQC_Fashion_{classf_mode}clas_{N}qub_{DEPTH}dep.pdparams")
            net.set_state_dict(net_state_dict)
        elif model == 'QCL':
            net = QCL(n=N, depth=DEPTH)
            net_state_dict = paddle.load(f"Model/QCL_Fashion_{classf_mode}clas_{N}qub_{DEPTH}dep.pdparams")
            net.set_state_dict(net_state_dict)
        elif model == 'QCNN':
            net = QCNN(n=N)
            net_state_dict = paddle.load(f"Model/QCNN_Fashion_{classf_mode}clas_{N}qub.pdparams")
            net.set_state_dict(net_state_dict)
        else:
            raise Exception("Model does not exist.")
    else:
        raise Exception("Dataset does not exist.")
        
    return net 