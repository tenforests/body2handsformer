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
	def __init__(self,hand_input=42*2,body_input=6*3,joint=48,t_out_dim=512,s_out_dim=64,
		        nhead = 8, dropout = 0.2, T_num_encoder_layers = 6,
			  	  S_num_encoder_layers = 6, feedforward_dim = 2048):
		super(ST_former, self).__init__()
		self.hand_input = hand_input
		self.body_input = body_input
		self.nhead = nhead
		self.t_out_dim = t_out_dim
		self.s_out_dim = s_out_dim
		self.joint = joint
		self.dropout = dropout
		self.S_num_encoder_layers = S_num_encoder_layers
		self.T_num_encoder_layers = T_num_encoder_layers
		self.feedforward_dim = feedforward_dim
	
	def build_net(self,batch_size=10*32):
		self.batch_size = batch_size
		self.T_input_embdding = nn.Sequential(nn.Dropout(self.dropout),
			nn.Linear(self.hand_input+self.body_input, self.t_out_dim),
			nn.LeakyReLU(0.2, True),
			nn.LayerNorm(self.t_out_dim))
		
		self.S_input_embdding = nn.Sequential(nn.Dropout(self.dropout),
				nn.Linear(3, self.s_out_dim),
				nn.LeakyReLU(0.2, True),
				nn.LayerNorm(self.s_out_dim))
		# 定义可学习的位置编码
		self.T_pos_embed = nn.Parameter(torch.zeros(self.batch_size, self.t_out_dim))
		self.S_pos_embed = nn.Parameter(torch.zeros(1,self.joint+1,self.s_out_dim))
		# 定义编码层
		self.S_encoder_layer = nn.TransformerEncoderLayer(d_model=self.s_out_dim, nhead=self.nhead,
						    dim_feedforward=self.feedforward_dim,dropout=self.dropout)
		self.T_encoder_layer = nn.TransformerEncoderLayer(d_model=self.t_out_dim, nhead=self.nhead,
						    dim_feedforward=self.feedforward_dim,dropout=self.dropout)
		# 定义 transformer 编码器 
		self.S_transformer = nn.TransformerEncoder(self.S_encoder_layer, 
					     num_layers=self.S_num_encoder_layers, mask_check=True)
		self.T_transformer = nn.TransformerEncoder(self.T_encoder_layer, 
					     num_layers=self.T_num_encoder_layers, mask_check=True)
		#
		self.out = nn.Sequential(
			nn.LayerNorm(self.s_out_dim),
			nn.Linear(self.s_out_dim, self.t_out_dim))
		self.regression_head = nn.Sequential(
            nn.LayerNorm(self.t_out_dim),
            nn.Linear(self.t_out_dim , 48*3))
	
	def Temporal_forward(self,hand_input,body_input):    # 每次对一个batch的数据进行处理  (n*f)*(42*2)   (n*f)*(6*3)
		# (n*f) * (42*2) (6*3)
		T_input = torch.concat((hand_input,body_input),dim=1)        # (n*f)*102
		x = self.T_input_embdding(T_input)                           # (n*f)*512
		x = x.add(self.T_pos_embed)                                  # 加入位置编码
		out = self.T_transformer(x)                                  
		return out

	def Spatial_forward(self,hand_input,body_input):	 # 每次对一个batch的数据进行处理  (n*f)*(42*2)   (n*f)*(6*3)
		# frame*feature (42*2) (6*3)
		hand_input = rearrange(hand_input, 'f (w c) -> f w c', w=42)     # (n*f)*42*2
		body_input = rearrange(body_input, 'f (w c) -> f w c', c=3)      # (n*f)*6*3
		zero_tensor= torch.zeros((self.batch_size,42,1))
		hand_input = torch.cat((hand_input,zero_tensor),2)               # (n*f)*42*3
		s_input = torch.cat((hand_input,body_input),1)				     # (n*f)*48*3
		x = self.S_input_embdding(s_input)                               # (n*f)*48*64
		res = torch.empty((0,self.joint+1,self.s_out_dim))               # 记录最终的输出
		self.class_token = nn.Parameter(torch.randn(self.batch_size,1,self.s_out_dim))  
		# for batch in x:                                                # 一帧一帧处理
			# batch = batch.reshape(-1,self.joint,self.s_out_dim)        # 1*joint*64
			# batch = torch.concat((batch,self.class_token),dim=1)		 # 1*(joint+1)*64
			# #print("size:",batch.shape)
			# batch = batch.add(self.S_pos_embed)                        # 加入位置编码
			# out = self.S_transformer(batch)                            # 每帧的关节点作为输入
			# res = torch.cat((res,out),0)
		res = torch.concat((self.class_token,x),dim=1)           	     # (n*f)*49*64
		res = res[:,0,:]												 # (n*f)*(joint+1)*64 -> (n*f)*64
		res = self.out(res)											     # (n*f)*64    -> (n*f)*512
		return res
	
	def forward(self,hand_input,body_input):                              # (n*f)*(42*2)   (n*f)*(6*3)
		batch,_=hand_input.shape
		self.batch_size = batch
		s_out = self.Temporal_forward(hand,body)                          # 
		t_out = self.Spatial_forward(hand,body)
		_out = s_out.add(t_out)    			   					     	  # (n*f)*512
		_out = self.regression_head(_out)                            	  # (n*f)*(48*3)
		#print("size:",_out.shape)
		#_out = _out.reshape(1,self.seq_length,48*3)
		#print("size:",_out.shape,res.shape)
		#res = torch.cat((res,_out),dim=0)
		res = rearrange(_out, 'f (a b) -> f a b', b=3)
		return res                          						  	  # (n*f)*48*3

S = ST_former()
S.build_net(10*32)
hand = torch.randn((10*32,42*2))
body = torch.randn((10*32,6*3))
x = S.forward(hand,body)
print(x.shape)