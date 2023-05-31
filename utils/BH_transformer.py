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
# to fix
class BH_transformer(nn.Module):
	def __init__(self):
		super(BH_transformer, self).__init__()
	
	def build_net(self, body_input_dim=8*3,hand_input_dim=15*3,two_hand_dim=720,input_in_dim=360,out_dim=90,
	       seq_length = 32,nhead =9, dropout = 0.5,num_encoder_layers =3,num_decoder_layers = 3, 
		   dim_feedforward = 2048):
		
# 定义 embedding 
		
		self.body_embedding = nn.Sequential(nn.Dropout(dropout),
					nn.Linear(body_input_dim, input_in_dim),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(seq_length, momentum=0.01))
		self.left_hand_embedding = nn.Sequential(nn.Dropout(dropout),
					nn.Linear(hand_input_dim, input_in_dim),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(seq_length, momentum=0.01))
		self.right_hand_embedding = nn.Sequential(nn.Dropout(dropout),
					nn.Linear(hand_input_dim, input_in_dim),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(seq_length, momentum=0.01))


		# 定义编码层
		self.body_encoder_layer = nn.TransformerEncoderLayer(d_model=input_in_dim, nhead=nhead,
						    dim_feedforward=dim_feedforward,dropout=dropout)
		self.left_hand_decoder_layer = nn.TransformerDecoderLayer(d_model=input_in_dim, nhead=nhead,
						    dim_feedforward=dim_feedforward,dropout=dropout)
		self.right_hand_decoder_layer = nn.TransformerDecoderLayer(d_model=input_in_dim, nhead=nhead,
						    dim_feedforward=dim_feedforward,dropout=dropout)
		self.two_hand_decoder_layer = nn.TransformerDecoderLayer(d_model=input_in_dim, nhead=nhead,
						    dim_feedforward=dim_feedforward,dropout=dropout)
		

		# 定义 transformer 
		self.body_encoder=nn.TransformerEncoder(self.body_encoder_layer,num_layers=num_encoder_layers)
		
		self.left_hand_decoder=nn.TransformerDecoder(self.left_hand_decoder_layer,num_layers=num_decoder_layers)
		
		self.right_hand_decoder=nn.TransformerDecoder(self.right_hand_decoder_layer,num_layers=num_decoder_layers)
		
		self.two_hand_decoder=nn.TransformerDecoder(self.two_hand_decoder_layer,num_layers=num_decoder_layers)
		
		#定义MLP 720D->360D
		self.mlp=nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(two_hand_dim, 512),
            nn.LeakyReLU(0.2, True),
	    	nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 360)
		)
		
        
   
		# 定义 regression   360D->90D 252d
		self.out = nn.Sequential(nn.Dropout(dropout),
				nn.Linear(input_in_dim, out_dim))
	


	def forward(self,body_input,left_hand_input,right_hand_input):
		 # 24  45  45
		
		body_input_in=self.body_embedding(body_input)                      # 将输入的身体特征 N*T*24D->N*T*360D   
		left_hand_input_in=self.left_hand_embedding(left_hand_input)       # 将输入的左手特征 N*T*45D->N*T*360D
		right_hand_input_in=self.right_hand_embedding(right_hand_input)    # 将输入的右手特征 N*T*45D->N*T*360D
	    
        #body_output Q1
		body_output=self.body_encoder(body_input_in)
	
		left_hand_output=self.left_hand_decoder(body_output,left_hand_input_in)
		right_hand_output=self.right_hand_decoder(body_output,right_hand_input_in)
	    
        #左右手级联 ver1.0
		hand_input=torch.concat((left_hand_output,right_hand_output),dim=2)   #360D->720D
	    #MLP
		hand_input_in=self.mlp(hand_input)              #720D->360D

	    
		x=self.two_hand_decoder(body_input_in,hand_input_in)
		out=self.out(x)
	   
		return out
	
class autoEncoder(nn.Module):
	def init(self):
		super(autoEncoder,self).__init__()
	
	def build_net(self,hand_in_feature:21*6,emb_feature:32,dropout:0.1):
		self.encoder = nn.Sequential(
					nn.Dropout(dropout),
					nn.Linear(hand_in_feature,64),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(64, momentum=0.01),
					nn.Dropout(dropout),
					nn.Linear(64,emb_feature)
		)
		self.decoder = nn.Sequential(
					nn.Dropout(dropout),
					nn.Linear(emb_feature,64),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(64, momentum=0.01),
					nn.Dropout(dropout),
					nn.Linear(64,hand_in_feature)
		)
	def forward(self,input):
		emb = self.encoder(input)
		return emb

	def embout(self,emb):
		out = self.decoder(emb)
		return out

class body2handMLP(nn.Module):
	def init(self):
		super(body2handMLP,self).__init__()
	def build_net(self,body_in_feature:6*6,emb_feature:32,dropout:0.1,l_hand_encoder,r_hand_encoder):		
		self.l_hand_encoder = l_hand_encoder
		self.r_hand_encoder = r_hand_encoder
		self.l_MLP = nn.Sequential(
					nn.Dropout(dropout),
					nn.Linear(body_in_feature,64),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(64, momentum=0.01),
					nn.Dropout(dropout),
					nn.Linear(64,emb_feature)
		)
		self.r_MLP = nn.Sequential(
					nn.Dropout(dropout),
					nn.Linear(body_in_feature,64),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(64, momentum=0.01),
					nn.Dropout(dropout),
					nn.Linear(64,emb_feature)
		)
	def forward(self,input):
		l_emb = self.l_MLP(input)
		r_emb = self.r_MLP(input)
		lout = self.l_hand_encoder.embout(l_emb)
		rout = self.r_hand_encoder.embout(r_emb)
		return l_emb,r_emb,lout,rout
# S = BH_transformer()
# S.build_net()
# body_input=torch.randn((10,32,24))
# left_hand_input=torch.randn((10,32,45))
# right_hand_input=torch.randn((10,32,45))

# x = S.forward(body_input,left_hand_input,right_hand_input)
# print(x.shape)    
		