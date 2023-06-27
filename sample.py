# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import json
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from einops import rearrange, repeat

import utils.ST_former_2 as ST_former
from utils.load_utils import *
from utils.mertic import *

DATA_PATHS = {
        #'video_data/Test/Oliver/':1,
        # 'video_data/Test/Chemistry/':2,
        # 'video_data/Test/Seth/':5,
        #'video_data/Test/Almaram/':3,
        # 'video_data/Test/Angelica/':4,
        #'video_data/Test/Ellen/':5,
        # 'video_data/Test/Rock/':7,
        #'video_data/Test/Shelly/':8,
        #'video_data/Test/Conan/':6,
        'video_data/Multi/sample/':9
        }

def main(args):
    ## variable initializations
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    pipeline = args.pipeline
    feature_in_dim, feature_out_dim = FEATURE_MAP[pipeline]
    feats = pipeline.split('2')
    in_feat, out_feat = feats[0], feats[1]
    ## DONE variable initializations


    ## set up model/ load pretrained model
    args.model = 'ST_former'
    model = getattr(ST_former,args.model)(hand_input=args.hand_input, body_input=args.body_input, t_out_dim=args.t_out_dim, s_out_dim=args.s_out_dim,
		        nhead = args.nhead, dropout = args.dropout, 
				T_num_encoder_layers = args.T_num_decoder_layers, S_num_encoder_layers = args.S_num_decoder_layers,
                S_feedforward_dim = args.S_feedforward_dim,T_feedforward_dim = args.T_feedforward_dim)
    model.build_net(args.seq_length)
    pretrain_model = args.checkpoint
    loaded_state = torch.load(pretrain_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(loaded_state['state_dict'], strict=False)
    model = model.eval()
    model.to(device)
    criterion = nn.L1Loss()
    ## DONE set up model/ load pretrained model

    gt_windows = None
    quant_windows = None
    p0_paths = None
    hand_ims = None
    ## load/prepare data from external files
    #test_X, test_Y, test_Y_paths, _ = load_windows(args.data_dir, args.pipeline, require_image=args.require_image)
    ######################
    for key, value in DATA_PATHS.items():
        key = os.path.join(args.data_dir, key)
        #print("Hello:",key)
        # p0 身体 p1 手部特征 paths 路径
        #curr_p0, curr_p1, curr_paths, _
        test_X, test_Y, test_Y_paths, _ = load_windows(key, args.pipeline, require_image=args.require_image,frame=args.seq_length)
        #print("test:",test_X[0].shape,test_X[1].shape,args.seq_length)
        if gt_windows is None:
            if args.require_image:
                hand_ims = test_X[1]
                test_X = test_X[0]

            gt_windows = test_X
            quant_windows = test_Y
            p0_paths = test_Y_paths
        else:
            if args.require_image:
                print("shape:",hand_ims.shape,test_X[1].shape)
                hand_ims = np.concatenate((hand_ims, test_X[1]), axis=0)     # 1 装的是手部6D表示
                test_X = test_X[0]
            gt_windows = np.concatenate((gt_windows, test_X), axis=0)        # 0 装的是身体6D表示
            quant_windows = np.concatenate((quant_windows, test_Y), axis=0)
            p0_paths = np.concatenate((p0_paths, test_Y_paths), axis=0)
   ######################################         


    test_X = np.swapaxes(gt_windows, 1, 2).astype(np.float32) 
    test_Y = np.swapaxes(quant_windows, 1, 2).astype(np.float32)

    # standardize
    checkpoint_dir = os.path.split(pretrain_model)[0]
    # model_tag = os.path.basename(args.checkpoint).split(args.pipeline)[0]
    model_tag=args.checkpoint.split('/')[1]
    preprocess = np.load(os.path.join("checkpoint/",'{}{}_preprocess_core.npz'.format(model_tag, args.pipeline)))
    body_mean_X = preprocess['body_mean_X']
    body_std_X = preprocess['body_std_X']
    body_mean_Y = preprocess['body_mean_Y']
    body_std_Y = preprocess['body_std_Y']
    test_X = (test_X - body_mean_X) / body_std_X
    test_Y = (test_Y - body_mean_Y) / body_std_Y
    ## DONE load/prepare data from external files

    test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) 
    test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32)


    ## pass loaded data into training
    inputData = Variable(torch.from_numpy(test_X)).to(device)
    outputGT = Variable(torch.from_numpy(test_Y)).to(device)
    imsData = None
    if args.require_image:
        imsData = Variable(torch.from_numpy(hand_ims)).to(device)
    # inputData = np.swapaxes(inputData, 1, 2)
    # print("check:",inputData.shape,imsData.shape)

    batchinds = np.arange(inputData.shape[0] // args.batch_size)
    print("size:",inputData.shape[0],args.batch_size)
    totalSteps = len(batchinds)
    print("batch_:",batchinds)
    sum_L1,sum_MPJPE,sum_MPJRE=0,0,0
    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * args.batch_size
        inputData_some = inputData[idxStart:(idxStart + args.batch_size), :, :]
        outputGT_some = outputGT[idxStart:(idxStart + args.batch_size), :, :]
        imsData_some = imsData[idxStart:(idxStart + args.batch_size), :, :]
        inputData_some=inputData_some[:,:64*12,:].reshape(-1,64,36)
        imsData_some=imsData_some[:,:64*12,:].reshape(-1,64,1024)
        outputGT_some=outputGT_some[:,:64*12,:].reshape(-1,64,252)
        output_some = model(body_input=inputData_some,hand_input=imsData_some.float())
        error = criterion(output_some, outputGT_some).data
        MPJRE_loss,MPJPE_loss = evaluate_prediction(sample=output_some,gt_sample=outputGT_some,device=device,mean=torch.from_numpy(body_mean_Y).to(device),
                                            std=torch.from_numpy(body_std_Y).to(device),already_std=False,reverse=True,is_adam=True)
        sum_L1+=error
        sum_MPJPE+=MPJPE_loss
        sum_MPJRE+=MPJRE_loss
    sum_L1/=totalSteps
    sum_MPJPE/=totalSteps
    sum_MPJRE/=totalSteps
    # inputData = np.swapaxes(inputData, 1, 2)
    # MPJRE_loss,MPJPE_loss = evaluate_prediction(sample=output,gt_sample=outputGT,device=args.device,mean=torch.from_numpy(args.body_mean_Y).to(args.device),std=torch.from_numpy(args.body_std_Y).to(args.device),already_std=False)
    # print(">>> L1_Loss MPJPE MPJRE: ", error,MPJRE_loss,MPJPE_loss)
    print('----------------------------------')
    ## DONE pass loaded data into training


    ## preparing output for saving
    # output_np = output.data.cpu().numpy()
    # output_gt = outputGT.data.cpu().numpy()
    # output_np = output_np * body_std_Y + body_mean_Y
    # output_gt = output_gt * body_std_Y + body_mean_Y
    # output_np = np.swapaxes(output_np, 1, 2).astype(np.float32)
    # output_gt = np.swapaxes(output_gt, 1, 2).astype(np.float32)
    # output_np = torch.tensor(output_np).to(args.device)
    # output_gt = torch.tensor(output_gt).to(args.device)
    

    print(">>> L1_Loss %.5f MPJRE %.5f MPJPE %.5f " %(sum_L1,sum_MPJRE,sum_MPJPE))
    # save_results(test_Y_paths, output_np, args.pipeline, args.base_path, tag=args.tag+str(error))
    ## DONE preparing output for saving


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint file (pretrained model)')
    parser.add_argument('--base_path', type=str, required=True, help='absolute path to the base directory where all of the data is stored')
    parser.add_argument('--data_dir', type=str, required=True, help='path to test data directory')
    parser.add_argument("--cuda",type=str,default = 'cuda:0',help = 'gpu use')
    parser.add_argument('--pipeline', type=str, default='arm2wh', help='pipeline specifying which input/output joints to use')
    parser.add_argument('--require_image', action='store_true', help='step size for prining log info')
    parser.add_argument('--tag', type=str, default='', help='prefix for naming purposes')
    parser.add_argument("--seq_length",type=int , default=32, help='frame')
    parser.add_argument("--hand_input",type=int , default=1024, help='hand feature')
    parser.add_argument("--body_input",type=int , default=6*6, help='body input size')
    parser.add_argument("--t_out_dim",type=int , default=512, help='temporal output')
    parser.add_argument("--s_out_dim",type=int , default=64, help='spatil output')
    parser.add_argument("--nhead",type=int , default=8, help='multi head att num(must can divide input dim)')
    parser.add_argument("--dropout",type=float , default=0.1, help='pos emb dropout rate')
    parser.add_argument("--S_num_decoder_layers",type=int , default=6, help='decoder layer number')
    parser.add_argument("--T_num_decoder_layers",type=int , default=6, help='decoder layer number')
    parser.add_argument("--S_feedforward_dim",type=int , default=128, help='fc node num')
    parser.add_argument("--T_feedforward_dim",type=int , default=1024, help='fc node num')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    args = parser.parse_args()
    print(args)
    main(args)
    