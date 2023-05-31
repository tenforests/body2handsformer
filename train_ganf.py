# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import numpy as np
import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
# from torchsummary import summary
import utils.modelZoo as modelZoo
from utils.modelZoo import PositionalEncoding
from utils.load_utils import *
import einops

DATA_PATHS = {
        'video_data/Oliver/train/':1,
        'video_data/Chemistry/train/':2,
        'video_data/Seth/train/':5,
        # 'video_data/Almaram/train':3,
        # 'video_data/Angelica/train':4,
        # 'video_data/Ellen/train':5,
        # 'video_data/Rock/train':7,
        # 'video_data/Shelly/train':8,
        'video_data/Conan/train/':6,
        }


#######################################################
## main training function
#######################################################
def main(args):
    ## variables
    learning_rate = args.learning_rate
    pipeline = args.pipeline
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('---------------------------------',device,'-------------------------------')
    feature_in_dim, feature_out_dim = FEATURE_MAP[pipeline]
    feats = pipeline.split('2')
    in_feat, out_feat = feats[0], feats[1]
    currBestLoss = 1e3
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    ## DONE variables
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    ## set up generator model
    args.model = 'body2handformer'
    generator = getattr(modelZoo,args.model)()
    generator.build_net(feature_in_dim=36,  feature_out_dim=args.feature_out_dim, hand_dim = 252,
	       seq_length = args.seq_length,nhead =args.nhead, dropout = args.dropout,
		   num_encoder_layers =args.num_encoder_layers,num_decoder_layers =args.num_decoder_layers, 
		   dim_feedforward = args.dim_feedforward,require_image=args.require_image,pos_embedding = PositionalEncoding(args.feature_out_dim,args.pos_dropout,args.seq_length),
		    mem_mask = args.mem_mask,output_mask = args.output_mask)
    # pretrain_model = args.checkpoint
    # loaded_state = torch.load(pretrain_model, map_location=lambda storage, loc: storage)
    # generator.load_state_dict(loaded_state['state_dict'], strict=False)
    # generator = getattr(modelZoo, args.model)()
    # generator.build_net(feature_in_dim, feature_out_dim, require_image=args.require_image)
    generator.cuda()
    reg_criterion = nn.L1Loss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-5)
    generator.train()
    ## set up discriminator model
    args.model = 'regressor_fcn_bn_discriminator'
    discriminator = getattr(modelZoo, args.model)()
    discriminator.build_net(feature_out_dim)
    discriminator.cuda()
    gan_criterion = nn.MSELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)
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



#######################################################
## local helper methods
#######################################################

## function to load data from external files
def load_data(args, rng):
    gt_windows = None
    quant_windows = None
    p0_paths = None
    hand_ims = None

    ## load from external files
    for key, value in DATA_PATHS.items():
        key = os.path.join(args.base_path, key)
        curr_p0, curr_p1, curr_paths, _ = load_windows(key, args.pipeline, require_image=args.require_image)
        if gt_windows is None:
            if args.require_image:
                hand_ims = curr_p0[1]
                curr_p0 = curr_p0[0]

            gt_windows = curr_p0
            quant_windows = curr_p1
            p0_paths = curr_paths
        else:
            if args.require_image:
                hand_ims = np.concatenate((hand_ims, curr_p0[1]), axis=0)
                curr_p0 = curr_p0[0]
            gt_windows = np.concatenate((gt_windows, curr_p0), axis=0)
            quant_windows = np.concatenate((quant_windows, curr_p1), axis=0)
            p0_paths = np.concatenate((p0_paths, curr_paths), axis=0)

    print("====>  in/out", gt_windows.shape, quant_windows.shape)
    if args.require_image:
         print("====> hand_ims", hand_ims.shape)
    ## DONE load from external files


    ## shuffle and set train/validation
    N = gt_windows.shape[0]
    train_N = int(N * 0.7)
    idx = np.random.permutation(N)
    train_idx, test_idx = idx[:train_N], idx[train_N:]
    train_X, test_X = gt_windows[train_idx, :, :], gt_windows[test_idx, :, :]
    train_Y, test_Y = quant_windows[train_idx, :, :], quant_windows[test_idx, :, :]
    if args.require_image:
        train_ims, test_ims = hand_ims[train_idx,:,:], hand_ims[test_idx,:,:]
        train_ims = train_ims.astype(np.float32)
        test_ims = test_ims.astype(np.float32)
# 时间，特征 ->特征，时间
    train_X = np.swapaxes(train_X, 1, 2).astype(np.float32)
    train_Y = np.swapaxes(train_Y, 1, 2).astype(np.float32)
    test_X = np.swapaxes(test_X, 1, 2).astype(np.float32)
    test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32)

    body_mean_X, body_std_X, body_mean_Y, body_std_Y = calc_standard(train_X, train_Y, args.pipeline)
    np.savez_compressed(args.model_path + '{}{}_preprocess_core.npz'.format(args.tag, args.pipeline), 
            body_mean_X=body_mean_X, body_std_X=body_std_X,
            body_mean_Y=body_mean_Y, body_std_Y=body_std_Y) 

    train_X = (train_X - body_mean_X) / body_std_X
    test_X = (test_X - body_mean_X) / body_std_X
    train_Y = (train_Y - body_mean_Y) / body_std_Y
    test_Y = (test_Y - body_mean_Y) / body_std_Y

    train_X = np.swapaxes(train_X, 1, 2).astype(np.float32)
    train_Y = np.swapaxes(train_Y, 1, 2).astype(np.float32)
    test_X = np.swapaxes(test_X, 1, 2).astype(np.float32)
    test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32)
    # if args.seq_length!=64:
    #     einops.rearrange(train_X,'b (c w1) f -> (b w1) c f',c = args.seq_length )
    #     einops.rearrange(train_Y,'b (c w1) f -> (b w1) c f',c = args.seq_length )
    #     einops.rearrange(test_X,'b (c w1) f -> (b w1) c f',c = args.seq_length )
    #     einops.rearrange(test_Y,'b (c w1) f -> (b w1) c f',c = args.seq_length )
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
        inputData = Variable(torch.from_numpy(inputData_np)).cuda()
        outputGT = Variable(torch.from_numpy(outputData_np)).cuda()

        imsData = None
        if args.require_image:
            imsData_np = train_ims[idxStart:(idxStart + args.batch_size), :, :]
            imsData = Variable(torch.from_numpy(imsData_np)).cuda()
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


## training generator function
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
        inputData = Variable(torch.from_numpy(inputData_np)).cuda()
        outputGT = Variable(torch.from_numpy(outputData_np)).cuda()

        imsData = None
        if args.require_image:
            imsData_np = train_ims[idxStart:(idxStart + args.batch_size), :, :]
            imsData = Variable(torch.from_numpy(imsData_np)).cuda()
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


## validating generator function
def val_generator(args, generator, discriminator, reg_criterion,d_optimizer, g_optimizer, test_X, test_Y, currBestLoss, test_ims=None):
    testLoss = 0
    generator.eval()
    # discriminator.eval()
    batchinds = np.arange(test_X.shape[0] // args.batch_size)
    totalSteps = len(batchinds)
    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * args.batch_size
        inputData_np = test_X[idxStart:(idxStart + args.batch_size), :, :]
        outputData_np = test_Y[idxStart:(idxStart + args.batch_size), :, :]
        inputData = Variable(torch.from_numpy(inputData_np)).cuda()
        outputGT = Variable(torch.from_numpy(outputData_np)).cuda()

        imsData = None
        if args.require_image:
            imsData_np = test_ims[idxStart:(idxStart + args.batch_size), :, :]
            imsData = Variable(torch.from_numpy(imsData_np)).cuda()
        ## DONE setting batch data
        
        output = generator(seq_input=inputData, hand_input=outputGT,img_input=imsData)
        output = generator.out(output)
        output = output.contiguous().view(-1, output.size(-1))
        outputGT = outputGT.contiguous().view(-1, outputGT.size(-1))
        g_loss = reg_criterion(output, outputGT)
        testLoss += g_loss.item() * args.batch_size

    testLoss /= totalSteps * args.batch_size
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(args.epoch, args.num_epochs, bii, totalSteps, 
                                                                                            testLoss, 
                                                                                            np.exp(testLoss)))
    print('----------------------------------')
    # if testLoss < currBestLoss:
    prev_save_epoch = args.epoch
    checkpoint = {'epoch': args.epoch,
                    'state_dict': generator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict()}
    fileName = args.model_path + '/{}{}_checkpoint_e{}_loss{:.4f}.pth'.format(args.tag, args.pipeline, args.epoch, testLoss)
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
    parser.add_argument("--cuda",type=str,default = 'cuda:0',help = 'gpu use')
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
    # parser.add_argument('--checkpoint', type=str,  help='path to checkpoint file (pretrained model)')
    args = parser.parse_args()
    print(args)
    main(args)
