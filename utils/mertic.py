import math
import os
import random
import numpy as np
import torch
from manopth.manolayer import ManoLayer
from utils import utils_transform
from manopth import demo

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
# METERS_TO_MM = 10000.0

def MPJRE(
    predicted_angle,
    gt_angle,
):
    diff = gt_angle - predicted_angle
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error

# ? 是正确的吗
def MPJPE(
    predicted_position,
    gt_position
):
    pos_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))
    )
    return pos_error


# 是否可以改进 不算5个为0的指尖点
def evaluate_prediction(
    sample,
    gt_sample,
    reverse: True,
    alerady_std:False,
    is_adam: True,
    mean,
    std,
    vis:False
):
    sample = sample.reshape(-1,252,1)
    gt_sample = sample.reshape(-1,252,1)
    if not alerady_std:
        sample = sample * std +mean
        gt_sample = gt_sample * std + mean
    # b x t x 42 x 6 -> (bxtx42) x 6 -> (bxtx42) x 3 -> b x (tx42x3)
    motion_pred = sample.squeeze().cuda()
    predicted_angle = (
    utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach())
    .reshape(motion_pred.shape[0], -1)
    .float()
)
    gt = gt_sample.squeeze().cuda()
    gt_angle = (
    utils_transform.sixd2aa(gt.reshape(-1, 6).detach())
    .reshape(motion_pred.shape[0], -1)
    .float()
)
    mpjre = MPJRE(predicted_angle,gt_angle)
    
    predicted_angle = predicted_angle.reshape(-1,42,3)
    gt_angle = gt_angle.reshape(-1,42,3)
    # 先拆分左右手
    gt_left = gt_angle[:,:21,:]
    gt_right = gt_angle[:,21:,:]
    pre_left = predicted_angle[:,:21,:]
    pre_right = predicted_angle[:,21:,:]
    # 关节点映射数组
    if is_adam:
        index = [0,8,7,6,12,11,10,20,19,18,16,15,14,4,3,2]
    else:
        index = np.arange(16).tolist()
    gt_left = gt_left[:,index,:].reshape(-1,3*16)
    gt_right = gt_right[:,index,:].reshape(-1,3*16)
    pre_left = pre_left[:,index,:].reshape(-1,3*16)
    pre_right = pre_right[:,index,:].reshape(-1,3*16)

    mano_left = ManoLayer(side='left', mano_root='../manopth/mano/models', use_pca=False, flat_hand_mean=True)
    mano_right = ManoLayer(side = 'right' ,mano_root='../manopth/mano/models', use_pca=False, flat_hand_mean=True)

    # 使用manopth得到postion mano 模型 与adam 模型是相反的 因此如果是adam模型提取数据，使用反向
    if reverse:
        gt_left *=-1
        gt_right *= -1
        pre_left *= -1
        pre_right *= -1
    #     mano_left = ManoLayer(side='left', mano_root='../manopth/mano/models', use_pca=False, flat_hand_mean=True)
    #     mano_right = ManoLayer(side = 'right' ,mano_root='../manopth/mano/models', use_pca=False, flat_hand_mean=True)
    # else :
    #     mano_left = ManoLayer(side='left', mano_root='../manopth/mano/models', use_pca=False, flat_hand_mean=True)
    #     mano_right = ManoLayer(side = 'right' ,mano_root='../manopth/mano/models', use_pca=False, flat_hand_mean=True)

    shape = torch.zeros([gt_left.shape[0],10])

    _1,j_l_p =mano_left(pre_left,shape)
    _2,j_r_p =mano_right(pre_right,shape)
    _3,j_l_g =mano_left(gt_left,shape)
    _4,j_r_g = mano_right(gt_right,shape)


    predict_position = torch.concat((j_l_p,j_r_p),axis = 0).reshape(-1,3)
    gt_position = torch.concat((j_l_g,j_r_g),axis = 0).reshape(-1,3)

    mpjpe = MPJPE(predict_position,gt_position)

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

    return mpjre,mpjpe 