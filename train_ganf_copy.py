# Copyright (c) Facebook, Inc. and its affiliates.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import numpy as np
import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torchsummary import summary
import utils.modelZoo as modelZoo
from utils.modelZoo import PositionalEncoding
from utils.mertic import evaluate_prediction
from utils.load_utils import *
from torch.utils.tensorboard import SummaryWriter  
import einops

DATA_PATHS = {
        #'video_data/Oliver/train/':1,
        'video_data/Chemistry/train/':2,  #
        'video_data/Seth/train/':5, #
        # 'video_data/Almaram/train':3,
        # 'video_data/Angelica/train':4,
        # 'video_data/Ellen/train':5,
        'video_data/Rock/train':7,#
        'video_data/Shelly/train':8,#
        #'video_data/Conan/train/':6,
        }




#######################################################
## main training function
#######################################################
def main(args):
    ## variables
    learning_rate = args.learning_rate                       
    pipeline = args.pipeline                                                 #arm2wh
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu') #device=cpu
    print(device)
    args.device = device
    
    print('---------------------------------',args.device,'-------------------------------')
    feature_in_dim, feature_out_dim = FEATURE_MAP[pipeline]                 #36 512
    feats = pipeline.split('2')
    in_feat, out_feat = feats[0], feats[1]
    currBestLoss = 1e3
    rng = np.random.RandomState(23456)        
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    ## DONE variables
    if not os.path.exists(args.model_path):   #创建保存模型的文件夹  每轮训练生成.pth文件
        os.makedirs(args.model_path)

    ## set up generator model
    args.model = 'body2handformer'              
    generator = getattr(modelZoo,args.model)()              #创建生成器模型实例
    generator.build_net(feature_in_dim=36,  feature_out_dim=args.feature_out_dim, hand_dim = 252,       #调用build_net方法
	       seq_length = args.seq_length,nhead =args.nhead, dropout = args.dropout,
		   num_encoder_layers =args.num_encoder_layers,num_decoder_layers =args.num_decoder_layers, 
		   dim_feedforward = args.dim_feedforward,require_image=args.require_image,pos_embedding = PositionalEncoding(args.feature_out_dim,args.pos_dropout,args.seq_length),
		    mem_mask = args.mem_mask,output_mask = args.output_mask)
    # pretrain_model = args.checkpoint
    # loaded_state = torch.load(pretrain_model, map_location=lambda storage, loc: storage)
    # generator.load_state_dict(loaded_state['state_dict'], strict=False)
    # generator = getattr(modelZoo, args.model)()
    # generator.build_net(feature_in_dim, feature_out_dim, require_image=args.require_image)
    generator.to(args.device)
    reg_criterion = nn.L1Loss()                 #设置损失函数
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-5)    #使用Adam优化器来优化生成器模型的参数 生成器优化器
    generator.train()            #将生成器设置为训练模式
    ## set up discriminator model
    args.model = 'regressor_fcn_bn_discriminator'           #设置判别器模型架构
    discriminator = getattr(modelZoo, args.model)()
    discriminator.build_net(feature_out_dim)
    discriminator.to(args.device)

    gan_criterion = nn.MSELoss()            #设置判别器模型的损失函数 计算生成器和判别器的差异
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)  #设置判别器模型的优化器
    discriminator.train()
    ## DONE model


    ## load data from saved files
    data_tuple = load_data(args, rng)
    if args.require_image:
        train_X, train_Y, test_X, test_Y, train_ims, test_ims = data_tuple
    else:
        train_X, train_Y, test_X, test_Y = data_tuple
        train_ims, test_ims = None, None
    ## DONE: load data from saved files
    # summary(generator,[(36, 64),(64, 1024)],batch_size=1)
    ## training job
    kld_weight = 0.05
    prev_save_epoch = 0
    patience = 20
    for epoch in range(args.num_epochs):
        args.epoch = epoch
        # train discriminator
        if epoch > 100 and (epoch - prev_save_epoch) > patience:
            print('early stopping at:', epoch)
            break

        # if epoch > 0 and epoch % 3 == 0:
        #     print('td')
        #     train_discriminator(args, rng, generator, discriminator, gan_criterion, d_optimizer, train_X, train_Y, train_ims=train_ims)
        # else:
        #     print('tg')
        train_generator(args, rng, generator, discriminator, reg_criterion, gan_criterion, g_optimizer, train_X, train_Y, train_ims=train_ims)
        currBestLoss = val_generator(args, generator, discriminator, reg_criterion,d_optimizer, g_optimizer, test_X, test_Y, currBestLoss, test_ims=test_ims)




#####################################帧率变化###########




#######################################################
## local helper methods
#######################################################

## function to load data from external files
def load_data(args, rng):               #从外部文件加载数据 并用训练集的均值方差对训练集和验证集的数据进行处理
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
            quant_windows = np.concatenate((quant_windows, curr_p1), axis=0)    # quant_windows 手部关节点
            p0_paths = np.concatenate((p0_paths, curr_paths), axis=0)             # p0_path 视频信息数据
        
    print("====>  in/out", gt_windows.shape, quant_windows.shape)
    if args.require_image:
         print("====> hand_ims", hand_ims.shape)
    ## DONE load from external files
    


    ## shuffle and set train/validation
    N = gt_windows.shape[0]       #N 样本数量
    train_N = int(N * 0.7)
    idx = np.random.permutation(N)    #生成随机排列的数组索引 0-  N-1
    train_idx, test_idx = idx[:train_N], idx[train_N:]                      #train_idx 训练集索引数组 test_idx测试集索引数组
    train_X, test_X = gt_windows[train_idx, :, :], gt_windows[test_idx, :, :]     #  train_X body关节点训练集  test_X body关节点测试集
    train_Y, test_Y = quant_windows[train_idx, :, :], quant_windows[test_idx, :, :]  #  train_Y hand关节点训练集 test_Y hand关节点测试集
    if args.require_image:
        train_ims, test_ims = hand_ims[train_idx,:,:], hand_ims[test_idx,:,:]   #train_ims 手部resnet训练集  test_ims 手部resnet测试集
        train_ims = train_ims.astype(np.float32)
        test_ims = test_ims.astype(np.float32)
  #调整输入数据维度
    train_X = np.swapaxes(train_X, 1, 2).astype(np.float32) # b*t*36->b*36*t
    train_Y = np.swapaxes(train_Y, 1, 2).astype(np.float32) # b*t*512->b*512*t
    test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) 
    test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32)

    #body_mean_X body关节点训练集均值 body_std_X body关节点训练集标准差 body_mean_Y hand关节点训练集均值 body_std_Y hand关节点训练集标准差                                                         
    body_mean_X, body_std_X, body_mean_Y, body_std_Y = calc_standard(train_X, train_Y, args.pipeline)   #body关节点 hand关节点 训练集
    args.Mean_Y,args.Std_Y=body_mean_Y,body_std_Y           #args.Mean_Y hand关节点训练集均值 args.Std_Y hand关节点训练集标准差
    np.savez_compressed(args.model_path + '{}{}_preprocess_core.npz'.format(args.tag, args.pipeline), 
            body_mean_X=body_mean_X, body_std_X=body_std_X,
            body_mean_Y=body_mean_Y, body_std_Y=body_std_Y)   #保存body hand 关节点训练集的均值和方差

    train_X = (train_X - body_mean_X) / body_std_X  # train_X body关节点训练集     用body关节点训练集均值标准差处理
    test_X = (test_X - body_mean_X) / body_std_X    # test_X body关节点测试集
    train_Y = (train_Y - body_mean_Y) / body_std_Y  # train_Y hand关节点训练集      用hand关节点训练集均值标准差处理
    test_Y = (test_Y - body_mean_Y) / body_std_Y    # test_Y hand关节点测试集

    train_X = np.swapaxes(train_X, 1, 2).astype(np.float32)    #b*36*t->b*t*36
    train_Y = np.swapaxes(train_Y, 1, 2).astype(np.float32)    #b*512*t->b*t*512
    test_X = np.swapaxes(test_X, 1, 2).astype(np.float32)     
    test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32)
    
    if args.seq_length!=64:    # b
        einops.rearrange(train_X,'b (c w1) f -> (b w1) c f',c = args.seq_length )
        einops.rearrange(train_Y,'b (c w1) f -> (b w1) c f',c = args.seq_length )
        einops.rearrange(test_X,'b (c w1) f -> (b w1) c f',c = args.seq_length )
        einops.rearrange(test_Y,'b (c w1) f -> (b w1) c f',c = args.seq_length )
    print ("====> train/test", train_X.shape, test_X.shape)
    print("=====> standardization done")

    # Data shuffle
    I = np.arange(len(train_X))
    rng.shuffle(I)
    train_X = train_X[I]
    train_Y = train_Y[I]
    if args.require_image:
        train_ims = train_ims[I]
        return (train_X, train_Y, test_X, test_Y, train_ims, test_ims)
    ## DONE shuffle and set train/validation

    return (train_X, train_Y, test_X, test_Y)


## calc temporal deltas within sequences
def calc_motion(tensor):
    res = tensor[:,:,:1] - tensor[:,:,:-1]
    return res


## training discriminator functin
def train_discriminator(args, rng, generator, discriminator, gan_criterion, d_optimizer, train_X, train_Y, train_ims=None):
    # device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    generator.eval()
    discriminator.train()
    batchinds = np.arange(train_X.shape[0] // args.batch_size)
    totalSteps = len(batchinds)
    rng.shuffle(batchinds)

    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * args.batch_size
        inputData_np = train_X[idxStart:(idxStart + args.batch_size), :, :]
        outputData_np = train_Y[idxStart:(idxStart + args.batch_size), :, :]
        inputData = Variable(torch.from_numpy(inputData_np)).to(args.device)
        outputGT = Variable(torch.from_numpy(outputData_np)).to(args.device)

        imsData = None
        if args.require_image:
            imsData_np = train_ims[idxStart:(idxStart + args.batch_size), :, :]
            imsData = Variable(torch.from_numpy(imsData_np)).to(args.device)
        ## DONE setting batch data

        with torch.no_grad():
            fake_data = generator(inputData, image_=imsData).detach()

        fake_motion = calc_motion(fake_data)
        real_motion = calc_motion(outputGT)
        fake_score = discriminator(fake_motion)
        real_score = discriminator(real_motion)

        d_loss = gan_criterion(fake_score, torch.zeros_like(fake_score)) + gan_criterion(real_score, torch.ones_like(real_score))
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()


## training generator function #
def train_generator(args, rng, generator, discriminator, reg_criterion, gan_criterion, g_optimizer, train_X, train_Y, train_ims=None):
    # discriminator.eval()
    generator.train()
    batchinds = np.arange(train_X.shape[0] // args.batch_size)
    totalSteps = len(batchinds)
    rng.shuffle(batchinds)
    avgLoss = 0.

    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * args.batch_size
        inputData_np = train_X[idxStart:(idxStart + args.batch_size), :, :]
        outputData_np = train_Y[idxStart:(idxStart + args.batch_size), :, :]
        inputData = Variable(torch.from_numpy(inputData_np)).to(args.device)
        outputGT = Variable(torch.from_numpy(outputData_np)).to(args.device)

        imsData = None
        if args.require_image:
            imsData_np = train_ims[idxStart:(idxStart + args.batch_size), :, :]
            imsData = Variable(torch.from_numpy(imsData_np)).to(args.device)
        ## DONE setting batch data

        output = generator(seq_input=inputData, hand_input=outputGT,img_input=imsData)
        output = generator.out(output)
        # fake_motion = calc_motion(output)
        # with torch.no_grad():
        #     fake_score = discriminator(fake_motion)
        # fake_score = fake_score.detach()
# + gan_criterion(fake_score, torch.ones_like(fake_score))
        # transformer原始损失修改版本
        # 将 output reshape=> batchsize*token, feacture 64*64,252
        output = output.contiguous().view(-1, output.size(-1))
        outputGT = outputGT.contiguous().view(-1, outputGT.size(-1))
        g_loss = reg_criterion(output, outputGT)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        avgLoss += g_loss.item() * args.batch_size
        
        if bii % args.log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(args.epoch, args.num_epochs, bii, totalSteps,
                                                                                          avgLoss / (totalSteps * args.batch_size), 
                                                                                          np.exp(avgLoss / (totalSteps * args.batch_size))))


## validating generator function    #生成器 判别器 损失函数 判别器优化器 生成器优化器  body测试集 hand测试集 最大Loss  手部resnet 
def val_generator(args, generator, discriminator, reg_criterion,d_optimizer, g_optimizer, test_X, test_Y, currBestLoss, test_ims=None):
    # device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    testLoss = 0
    generator.eval()   #生成器设置为评估模式
    # discriminator.eval()

      #新增部分开始
    total_mpjre = 0.  
    total_mpjpe = 0.
    num_samples = 0.
    #新增部分结束  s
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
                output = generator(seq_input=inputData, hand_input=total, img_input=imsData, inference = True)
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
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, MPJRE: {:.4f}, MPJPE: {:.4f}'.format(
        args.epoch, args.num_epochs, bii, totalSteps, testLoss, np.exp(testLoss), mpjre_mean, mpjpe_mean))
    #新增部分结束
    

    #tensorboard实例化
    writer=SummaryWriter('./Tensorlog/img1_ba64_r1e-3_ou512_sq64_e6d6_f1024_dr0_n8')
    writer.add_scalar('Loss',testLoss,args.epoch)
    writer.add_scalar('MPJRE',mpjre_mean,args.epoch)
    writer.add_scalar('MPJPE',mpjpe_mean,args.epoch)
    
    # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(args.epoch, args.num_epochs, bii, totalSteps, 
    #                                                                                         testLoss, 
    #                                                                                         np.exp(testLoss)))
    print('----------------------------------')
    # if testLoss < currBestLoss:
    prev_save_epoch = args.epoch
    checkpoint = {'epoch': args.epoch,
                    'state_dict': generator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict()}                  #
    fileName = args.model_path + '/{}{}_checkpoint_e{}_loss{:.4f}_MPJRE{:.4f}_MPJPE{:.4f}.pth'.format(args.tag, args.pipeline, args.epoch, testLoss,mpjre_mean,mpjpe_mean)
    torch.save(checkpoint, fileName)
    # checkpoint2 = {'epoch': args.epoch,
    #                 'state_dict': discriminator.state_dict(),
    #                 'd_optimizer':d_optimizer.state_dict()}
    # fileName2 =  args.model_path +'/DNET'+ '{}{}_checkpoint_e{}_loss{:.4f}.pth'.format(args.tag, args.pipeline, args.epoch, testLoss)
    # torch.save(checkpoint2,fileName2)
    currBestLoss = testLoss

    return currBestLoss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True, help='path to the directory where the data files are stored')
    parser.add_argument('--pipeline', type=str, default='arm2wh', help='pipeline specifying which input/output joints to use')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=4e-3, help='learning rate for training G and D')
    parser.add_argument('--require_image', action='store_true', help='use additional image feature or not')
    parser.add_argument('--model_path', type=str, required=True , help='path for saving trained models')
    parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')
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

    # parser.add_argument('--checkpoint', type=str,  help='path to checkpoint file (pretrained model)')
    args = parser.parse_args()
    print(args)
    main(args)
