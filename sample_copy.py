# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import json
import numpy as np
import torch
import torchvision
import einops
from torch import nn
from torch.autograd import Variable
import utils.modelZoo as modelZoo
from utils.modelZoo import PositionalEncoding
from utils.mertic import evaluate_prediction

import utils.modelZoo as modelZoo
from utils.load_utils import *

DATA_PATHS = {
        'video_data/Test/Almaram':1,
        'video_data/Test/Chemistry':2,
        'video_data/Test/Conan':3,
        'video_data/Test/Ellen':4,
        'video_data/Test/Oliver':5,
        'video_data/Test/Rock':6,
        'video_data/Test/Seth':7,
        #'video_data/Test/Shelly':8,
        }

def main(args):
    ## variable initializations
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu') #device=cpu
    args.device = device
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    pipeline = args.pipeline                #arm2wh
    feature_in_dim, feature_out_dim = FEATURE_MAP[pipeline]   # feature_in_dim=6*6 feature_out_dim=42*6
    
    ## DONE variable initializations


    ## set up model/ load pretrained model
    args.model = 'body2handformer'      #  modelZoo.py里的body2handformer模型  ####
    generator= getattr(modelZoo,args.model)()
    generator.build_net(feature_in_dim=36,  feature_out_dim=args.feature_out_dim, hand_dim = 252,       #调用build_net方法
	       seq_length = args.seq_length,nhead =args.nhead, dropout = args.dropout,
		   num_encoder_layers =args.num_encoder_layers,num_decoder_layers =args.num_decoder_layers, 
		   dim_feedforward = args.dim_feedforward,require_image=args.require_image,pos_embedding = PositionalEncoding(args.feature_out_dim,args.pos_dropout,args.seq_length),
		    mem_mask = args.mem_mask,output_mask = args.output_mask)


    pretrain_model = args.checkpoint                #预训练模型文件
    loaded_state = torch.load(pretrain_model, map_location=lambda storage, loc: storage)   #从该路径中加载预训练模型参数
    generator.load_state_dict(loaded_state['state_dict'], strict=False)                #加载到已创建的模型实例中
    generator = generator.eval()                #将模型切换到评估模式。在评估模式下，模型不进行梯度计算和反向传播
    generator.cuda()                    #将模型移动到 CUDA 设备上
    criterion = nn.L1Loss()           #定义损失函数为均方误差
    ## DONE set up model/ load pretrained model

    
    ## load/prepare data from external files        #test_X body数据  tesst_Y hand数据
    data_tuple = load_data(args)
    if args.require_image:
        test_X, test_Y,test_ims = data_tuple
    else:
        test_X, test_Y = data_tuple
        test_ims = None

    currBestLoss = val_generator(args, generator,criterion,test_X, test_Y,test_ims=test_ims)

 






def load_data(args):               #从外部文件加载数据 并用训练集的均值方差对训练集和验证集的数据进行处理
    gt_windows = None
    quant_windows = None
    p0_paths = None
    hand_ims = None

    ## load from external files
    for key, value in DATA_PATHS.items():              #遍历每个数据集的路径
        key = os.path.join(args.base_path, key)        #curr_p0 body+手部resnet curr_p1 hand数据 curr_paths 视频信息数据
        curr_p0, curr_p1, curr_paths, _ = load_windows(key, args.pipeline, require_image=args.require_image)
        if gt_windows is None:                  #初次创建
            if args.require_image:
                hand_ims = curr_p0[1]   #hand_ims 手部resnet
                curr_p0 = curr_p0[0]    #curr_p0  body数据

            gt_windows = curr_p0             #gt_windows body关节点
            quant_windows = curr_p1         #quant_windows hand关节点
            p0_paths = curr_paths         #p0_paths 视频信息数据
        else:                   #拼接数据集数据
            if args.require_image:
                hand_ims = np.concatenate((hand_ims, curr_p0[1]), axis=0)   #hand_img 手部resnet
                curr_p0 = curr_p0[0]
            gt_windows = np.concatenate((gt_windows, curr_p0), axis=0)       #gt_windows body关节点
            quant_windows = np.concatenate((quant_windows, curr_p1), axis=0)    # quant_windows hand关节点
            p0_paths = np.concatenate((p0_paths, curr_paths), axis=0)             # p0_path 视频信息数据
        
    print("====>  in/out", gt_windows.shape, quant_windows.shape)
    if args.require_image:
         print("====> hand_ims", hand_ims.shape)
    ## DONE load from external files
    

    if args.require_image:
        hand_ims = hand_ims.astype(np.float32)
  #调整输入数据维度
    gt_windows= np.swapaxes(gt_windows, 1, 2).astype(np.float32) # b*t*36->b*36*t
    quant_windows = np.swapaxes(quant_windows, 1, 2).astype(np.float32)  # b*t*512->b*512*t

    
        # standardize           #载入模型预处理的核心数据
    checkpoint_dir = os.path.split(args.checkpoint)[0]     #预训练所在目录
    model_tag = os.path.basename(args.checkpoint).split(args.pipeline)[0]        #从命令行参数中提取出模型标记 去掉arm2wh   
    #模型目录 模型tag arm2wh_preprocess_core.npz  npz文件在训练时已经生成 训练集的数据
    preprocess = np.load(os.path.join(checkpoint_dir,'{}{}_preprocess_core.npz'.format(model_tag, args.pipeline)))


    #对测试集数据进行标准化处理
    body_mean_X = preprocess['body_mean_X']      #body 均值
    body_std_X = preprocess['body_std_X']        #body 方差
    body_mean_Y = preprocess['body_mean_Y']     #hand 均值
    body_std_Y = preprocess['body_std_Y']       #hand方差

    gt_windows = (gt_windows - body_mean_X) / body_std_X   #对body数据进行标准化
    quant_windows = (quant_windows - body_mean_Y) / body_std_Y    #对hand数据进行标准化
    ## DONE load/prepare data from external files

    args.Mean_Y,args.Std_Y=body_mean_Y,body_std_Y           #args.Mean_Y hand关节点训练集均值 args.Std_Y hand关节点训练集标准差


    gt_windows = np.swapaxes(gt_windows, 1, 2).astype(np.float32)    #b*36*t->b*t*36
    quant_windows = np.swapaxes(quant_windows, 1, 2).astype(np.float32)    #b*512*t->b*t*512

    if args.seq_length!=64:    # b
        einops.rearrange(gt_windows,'b (c w1) f -> (b w1) c f',c = args.seq_length )
        einops.rearrange(quant_windows,'b (c w1) f -> (b w1) c f',c = args.seq_length )
        
    #print ("====> train/test", train_X.shape, test_X.shape)
    print("=====> standardization done")

    # Data shuffle
    if args.require_image:
        return (gt_windows,quant_windows,hand_ims)
    ## DONE shuffle and set train/validation

    return (gt_windows,quant_windows)



def val_generator(args, generator,reg_criterion,test_X,test_Y,test_ims=None):
    # device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    testLoss = 0
    generator.eval()   #生成器设置为评估模式

      #新增部分开始
    total_mpjre = 0.  
    total_mpjpe = 0.
    num_samples = 0.
    #新增部分结束  s
    # print('------shap:',test_X.shape)
                             #n=样本总数量/每次处理的样本数量 得到批次数量 [0,1,2...n)
    batchinds = np.arange(test_X.shape[0] // args.batch_size)
    totalSteps = len(batchinds)     #批次数量

    for bii, bi in enumerate(batchinds):   
        ## setting batch data
        idxStart = bi * args.batch_size     #当前样本序号
        inputData_np = test_X[idxStart:(idxStart + args.batch_size), :, :] #取出body验证集中 当前批次的所有样本   
        outputData_np = test_Y[idxStart:(idxStart + args.batch_size), :, :]  #取出hand验证集中 当前批次的所有样本  真实值
        inputData = Variable(torch.from_numpy(inputData_np)).to(args.device)   #转换为pytorch张量
        outputGT = Variable(torch.from_numpy(outputData_np)).to(args.device)

        imsData = None
        if args.require_image:
            imsData_np = test_ims[idxStart:(idxStart + args.batch_size), :, :]  #取出手部resnet验证集中 当前批次的所有样本
            imsData = Variable(torch.from_numpy(imsData_np)).to(args.device)
        ## DONE setting batch data
        with torch.no_grad():    #关闭梯度计算
            total = torch.tensor([]).to(args.device)
            for i in range(inputData.shape[1]):   #seq_length次
                output = generator(seq_input=inputData, hand_input=total, img_input=imsData, inference = True)  #
                # print(output.shape)
                output = generator.out(output[:,-1,:])
                output = einops.rearrange(output, 'b (t c) -> b t c', t = 1)  # 预测结果 b*1*(seq*output_dim)
                # print(output.shape)
                total = torch.concat([total,output],dim = 1)    #total 就包含了之前累积的输出和当前 output 的输出，形成了更长的序列。
                # print(total.shape)

            output = total.contiguous().view(-1, total.size(-1))  #(batchsize*seqlength)*output_dim
            # output = generator(seq_input=inputData, hand_input=outputGT,img_input=imsData)
            # output = generator.out(output)
            # output = output.contiguous().view(-1, output.size(-1))
            outputGT = outputGT.contiguous().view(-1, outputGT.size(-1))
            g_loss = reg_criterion(output, outputGT)
            testLoss += g_loss.item() 

            #新增部分开始
            mpjre, mpjpe = evaluate_prediction(output, outputGT, 
                                                    reverse=args.reverse, alerady_std=args.alerady_std, is_adam=args.is_adam,
                                                    mean=torch.from_numpy(args.Mean_Y).to(args.device), std=torch.from_numpy(args.Std_Y).to(args.device),device=args.device)
        total_mpjre += mpjre 
        total_mpjpe += mpjpe 
        # num_samples += outputGT.shape[0]
        #新增部分结束

    testLoss /= totalSteps

    #新增部分开始
    mpjre_mean = total_mpjre / totalSteps
    mpjpe_mean = total_mpjpe / totalSteps
    # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, MPJRE: {:.4f}, MPJPE: {:.4f}'.format(
    #     args.epoch, args.num_epochs, bii, totalSteps, testLoss, np.exp(testLoss), mpjre_mean, mpjpe_mean))
    #新增部分结束
    
    print('----------------------------------/n')
    print('L1Loss:%.5f,mpjre:%.5f,mpjpe:%.5f',(testLoss,mpjre,mpjpe))
   
    currBestLoss = testLoss

    return currBestLoss



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,default='/HDD_sdd/yl/project/body2handsformer/testmodel/testchock.pth', help='path to checkpoint file (pretrained model)') #
    parser.add_argument('--base_path', type=str, required=True, default='/HDD_sdd/zt/body2handDataset',help='path to the directory where the data files are stored')
    parser.add_argument('--pipeline', type=str, default='arm2wh', help='pipeline specifying which input/output joints to use')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=4e-3, help='learning rate for training G and D')
    parser.add_argument('--require_image', action='store_true', help='use additional image feature or not')
    parser.add_argument('--tag', type=str, default='', help='prefix for naming purposes')
    parser.add_argument("--device",type=str,default = 'cuda:0',help = 'gpu use')
    parser.add_argument("--feature_out_dim",type=int , default=512, help='transformer token size')
    parser.add_argument("--seq_length",type=int , default=64, help='token seq')
    parser.add_argument("--num_encoder_layers",type=int , default=6, help='encoder layer number')
    parser.add_argument("--num_decoder_layers",type=int , default=6, help='decoder layer number')
    parser.add_argument("--dim_feedforward",type=int , default=2048, help='fc node num')
    parser.add_argument("--dropout",type=float , default=0.1, help='dropout rate')
    parser.add_argument("--mem_mask",type=bool , default=False, help='encoder mem mask switch')
    parser.add_argument("--output_mask",type=bool , default=True, help='decoder input mask switch')
    parser.add_argument("--pos_dropout",type=float , default=0.1, help='pos emb dropout rate')
    parser.add_argument("--nhead",type=int , default=8, help='multi head att num(must can divide input dim)')
    parser.add_argument("--reverse",type=bool,default=True)
    parser.add_argument("--alerady_std",type=bool, default=False)
    parser.add_argument("--is_adam",type=bool,default=True)
    

    args = parser.parse_args()
    print(args)
    main(args)
