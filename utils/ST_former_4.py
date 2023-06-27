import os
import sys
import numpy as np
import scipy.io as io
import timm
import math
from einops import rearrange, repeat
import torch
import torchvision
from torch import nn
from torch.nn import Transformer
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from PIL import Image,ImageDraw 

#############################################################
#############  Temporal-Spatial  Transformer  ###############    
#############################################################

class ST_former(nn.Module):
	def __init__(self, hand_input=1024, body_input=6*6, t_out_dim=512, s_out_dim=64,
		        nhead = 8, dropout = 0.2, batch_size=10, seq_length =32, 
				T_num_encoder_layers = 6, S_num_encoder_layers = 6, 
				S_feedforward_dim = 128, T_feedforward_dim = 1024):
		super(ST_former, self).__init__()
		self.hand_input = hand_input
		self.body_input = body_input
		self.nhead = nhead
		self.t_out_dim = t_out_dim
		self.s_out_dim = s_out_dim
		self.dropout = dropout
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.S_num_encoder_layers = S_num_encoder_layers
		self.T_num_encoder_layers = T_num_encoder_layers
		self.S_feedforward_dim = S_feedforward_dim
		self.T_feedforward_dim = T_feedforward_dim
	
	def build_net(self):
		self.T_input_embdding_hand = nn.Sequential(nn.Dropout(self.dropout),
			nn.Linear(self.hand_input, self.t_out_dim//2),
			nn.LeakyReLU(0.2, True),
			nn.LayerNorm(self.t_out_dim//2))
		self.T_input_embdding_body = nn.Sequential(nn.Dropout(self.dropout),
			nn.Linear(self.body_input, self.t_out_dim//2),
			nn.LeakyReLU(0.2, True),
			nn.LayerNorm(self.t_out_dim//2))
		
		self.S_input_embdding = nn.Sequential(nn.Dropout(self.dropout),
				nn.Linear(6, self.s_out_dim),
				nn.LeakyReLU(0.2, True),
				nn.LayerNorm(self.s_out_dim))
		self.S_input_embdding_hand = nn.Sequential(nn.Dropout(self.dropout),
				nn.Linear(512, self.s_out_dim),
				nn.LeakyReLU(0.2, True),
				nn.LayerNorm(self.s_out_dim))
		
		#class_token的定义
		self.class_token = nn.Parameter(torch.randn(1,self.s_out_dim))
		# 定义可学习的位置编码
		self.T_pos_embed = nn.Parameter(torch.randn(self.seq_length, self.t_out_dim))
		self.S_pos_embed = nn.Parameter(torch.randn(8,self.s_out_dim))
		# 定义编码层
		self.T_encoder_layer = nn.TransformerEncoderLayer(d_model=self.t_out_dim, nhead=self.nhead,
						    dim_feedforward=self.T_feedforward_dim,dropout=self.dropout)
		self.S_encoder_layer = nn.TransformerEncoderLayer(d_model=self.s_out_dim, nhead=self.nhead,
						    dim_feedforward=self.S_feedforward_dim,dropout=self.dropout)
		# 定义 transformer 编码器 
		self.T_transformer = nn.TransformerEncoder(self.T_encoder_layer, 
					num_layers=self.T_num_encoder_layers, mask_check=True)
		self.S_transformer = nn.TransformerEncoder(self.S_encoder_layer, 
					     num_layers=self.S_num_encoder_layers, mask_check=True)
		#
		# self.out = nn.Sequential(nn.Dropout(self.dropout),
		# 	nn.Linear(self.s_out_dim, self.t_out_dim))
		self.regression_head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.t_out_dim+self.s_out_dim*8 , (self.t_out_dim+self.s_out_dim*8)*3),
	    	nn.LeakyReLU(0.2, True),
		    nn.LayerNorm((self.t_out_dim+self.s_out_dim*8)*3),
		    nn.Dropout(self.dropout),
            nn.Linear((self.t_out_dim+self.s_out_dim*8)*3,self.t_out_dim+self.s_out_dim*8),
	    	nn.LeakyReLU(0.2, True),
		    nn.LayerNorm(self.t_out_dim+self.s_out_dim*8),
		    nn.Dropout(self.dropout),
            nn.Linear(self.t_out_dim+self.s_out_dim*8,252)
			)
	
	def Temporal_forward(self,hand_input,body_input):  				 # 每次对一个 batch的数据进行处理 
		# n*f * (42*2) (6*3)
		#print("HERE:",hand_input.shape)
		hand_input_=self.T_input_embdding_hand(hand_input)
		body_input_=self.T_input_embdding_body(body_input)
		T_input = torch.concat((hand_input_,body_input_),dim=2)        # n*f*512
		#print("T:",T_input.shape,self.T_pos_embed.shape)
		x = T_input.add(self.T_pos_embed)                                  # 加入位置编码
		out = self.T_transformer(x)                                  
		return out  # n*f*512

	def Spatial_forward(self,hand_input,body_input):	 # 每次对(n*f)的数据并行处理  
		# n*f*(1024)   n*f*(6*6)
		#print("HERE:",hand_input.shape,body_input.shape)
		body_input = rearrange(body_input, 'n f (w c) ->(n f) w c', w=6)      # (n*f)*6*6
		x = self.S_input_embdding(body_input)                            # (n*f)*6*64
		hand_input = rearrange(hand_input,'n f (w c) ->(n f) w c', w=2)
		y = self.S_input_embdding_hand(hand_input)
		#y = self.S_input_embdding_hand(hand_input)
		# print("HERE:",x.shape,y.shape)
		#begin_token = self.class_token.expand((self.batch_size*self.seq_length,1,self.s_out_dim))
		res = torch.concat((y,x),dim=1)           	     # (n*f)*7*64
		res = res.add(self.S_pos_embed)                       			 # 加入位置编码
		res = self.S_transformer(res)									 # 送入S_transformer
		#res = res[:,0,:] # ((n*f)*1*512)								 # (n*f)*(joint+1)*64 -> (n*f)*64
		res = rearrange(res,'(n f) w c -> n f (w c)', n = self.batch_size)	 # n*f*512		
		#res = self.out(res)											     # n*f*64 -> n*f*512
		return res #n*f*(8*64)
	
	def forward(self,hand_input,body_input):                              # n*f*(1024)   n*f*(6*6)
		#print("HERR:",hand_input.shape,body_input.shape)
		t_out = self.Temporal_forward(hand_input,body_input)              # n*f*512
		s_out = self.Spatial_forward(hand_input,body_input)            			  # n*f*6*128   没有任何用处
		 #print("size:",s_out.shape)
		_out = torch.concat((s_out,t_out),2)    			   				#  n*f*(512+64)
		_out = self.regression_head(_out)                            	  #  n*f*(42*6)
		#print("size:",_out.shape)
		#_out = _out.reshape(1,self.seq_length,48*3)
		#print("size:",_out.shape,res.shape)
		#res = torch.cat((res,_out),dim=0)
		# res = rearrange(_out, 'n f (a b) -> n f a b', b=6)				  # n*f*42*6 
		return _out                          						  	  # 

# S = ST_former()
# S.build_net()
# hand = torch.randn((10,32,1024))
# body = torch.randn((10,32,6*6))
# x = S.forward(hand,body)
# print(x.shape)