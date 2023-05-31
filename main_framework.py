import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import utils.modelZoo as modelZoo
import utils.BH_transformer
import utils.ST_transformer
from pathlib import Path

def get_args():
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
        parser.add_argument("every_model_save" type = bool,default= True,help='control model save epoch,if set false,just save best model')
        parser.add_argument("modelname" type = str,default='body2handformer',help='modelname')
        args = parser.parse_args()
        print(args)
        return args

def main():
    device = torch.device(args.device)
    currBestLoss = 1e3
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    model = getattr(modelZoo,args.model)()
    model.build_net(feature_in_dim=36,  feature_out_dim=args.feature_out_dim, hand_dim = 252,
	       seq_length = args.seq_length,nhead =args.nhead, dropout = args.dropout,
		   num_encoder_layers =args.num_encoder_layers,num_decoder_layers =args.num_decoder_layers, 
		   dim_feedforward = args.dim_feedforward,require_image=args.require_image,pos_embedding = modelZoo.PositionalEncoding(args.feature_out_dim,args.pos_dropout,args.seq_length),
		    mem_mask = args.mem_mask,output_mask = args.output_mask)
    b2hdataset = 

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    main(args)
