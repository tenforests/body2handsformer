import math
import os
import random
import numpy as np
import torch
from manopth.manolayer import ManoLayer
from utils import utils_transform
#import utils_transform
import os
 



RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_MM = 1.0

def MPJRE(                                      #输入预测值和真实值的3D旋转坐标
    predicted_angle,
    gt_angle,
):
    diff = gt_angle - predicted_angle           
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error


def MPJPE(                                       #输入预测值和真实值的3维坐标（x,y,z)
    predicted_position,
    gt_position
):
    pos_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))
    )
    return pos_error



# 是否可以改进 不算5个为0的指尖点
def evaluate_prediction(
    sample,                 #预测数据
    gt_sample,              #真实数据
    reverse: True,         #是否使用反向的Mano模型reverse
    alerady_std:False,      #输入数据是否标准化
    is_adam: True,
    mean,                   #均值
    std,                    #标准差 
    device='cpu',
    vis:False
):
    sample = sample.reshape(-1,252,1)           #手部预测值
    gt_sample = gt_sample.reshape(-1,252,1)     #手部真实值
    if not alerady_std:                                     #如果预测数据被标准化过 进行一个逆向标准化
        sample = sample * std +mean
        gt_sample = gt_sample * std + mean
    # b x t x 42 x 6 -> (bxtx42) x 6 -> (bxtx42) x 3 -> b x (tx42x3)
    motion_pred = sample.squeeze().to(device)           #将张量中维度值为1的维度去掉 
    predicted_angle = (
    utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach())    #(b*t*42)*6->(b*t*42)*3   6D旋转->3D旋转
    .reshape(motion_pred.shape[0], -1)                      # (b*t*42)*3->b*(t*42*3)
    .float()
)
    gt = gt_sample.squeeze().to(device)
    gt_angle = (
    utils_transform.sixd2aa(gt.reshape(-1, 6).detach())
    .reshape(motion_pred.shape[0], -1)
    .float()
)
    mpjre = MPJRE(predicted_angle,gt_angle)*RADIANS_TO_DEGREES
    
    predicted_angle = predicted_angle.reshape(-1,42,3)     # b*(t*42*3)->(b*t)*42*3
    gt_angle = gt_angle.reshape(-1,42,3)
    # 先拆分左右手
    gt_left = gt_angle[:,:21,:]              #左手真实值
    gt_right = gt_angle[:,21:,:]            #右手真实值
    pre_left = predicted_angle[:,:21,:]     #左手预测值
    pre_right = predicted_angle[:,21:,:]        #右手预测值
    # 关节点映射数组
    if is_adam:
        index = [0,8,7,6,12,11,10,20,19,18,16,15,14,4,3,2]
    else:
        index = np.arange(16).tolist()
    gt_left = gt_left[:,index,:].reshape(-1,3*16)                       #(b*t)*42*3->(b*t)*(3*16) 左手真实值
    gt_right = gt_right[:,index,:].reshape(-1,3*16)
    pre_left = pre_left[:,index,:].reshape(-1,3*16)
    pre_right = pre_right[:,index,:].reshape(-1,3*16)

    mano_left = ManoLayer(side='left', mano_root='./mano/models', use_pca=False, flat_hand_mean=True).to(device)
    mano_right = ManoLayer(side = 'right' ,mano_root='./mano/models', use_pca=False, flat_hand_mean=True).to(device)

    if reverse:
        gt_left *=-1
        gt_right *= -1
        pre_left *= -1
        pre_right *= -1
    # 使用manopth得到postion mano 模型 与adam 模型是相反的 因此如果是adam模型提取数据，使用反向
    # if reverse:
    #     mano_left = ManoLayer(side='right', mano_root='./mano/models', use_pca=False, flat_hand_mean=True).to(device)
    #     mano_right = ManoLayer(side = 'left' ,mano_root='./mano/models', use_pca=False, flat_hand_mean=True).to(device)
    # else :
        
    #     mano_left = ManoLayer(side='left', mano_root='./mano/models', use_pca=False, flat_hand_mean=True).to(device)
    #     mano_right = ManoLayer(side = 'right' ,mano_root='./mano/models', use_pca=False, flat_hand_mean=True).to(device)

    shape = torch.zeros([gt_left.shape[0],10]).to(device)
    #print("device3:",pre_left.device,shape.device)
    _1,j_l_p =mano_left(pre_left,shape)          #  左手预测关节点
    _2,j_r_p =mano_right(pre_right,shape)        #   右手预测关节点
    _3,j_l_g =mano_left(gt_left,shape)           #   左手真实关节点
    _4,j_r_g = mano_right(gt_right,shape)        #   右手真实关节点
    # if vis  :
    #     os.makedies()
    if vis :
        os.makedirs("plts/right/")
        os.makedirs("plts/left/")

        demo.batch_display_hand({
    'verts': _1,
    'joints': j_l_p
},
                  mano_faces=mano_left.th_faces)
        
        demo.batch_display_hand({
    'verts': _2,
    'joints': j_r_p
},title = 'right',
                  mano_faces=mano_right.th_faces)
        
        
    predict_position = torch.concat((j_l_p,j_r_p),axis = 0).reshape(-1,3)    # 预测手部关节点
    gt_position = torch.concat((j_l_g,j_r_g),axis = 0).reshape(-1,3)         # 真实手部关节点

    mpjpe = MPJPE(predict_position,gt_position)

    return mpjre,mpjpe
