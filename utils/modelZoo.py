# Copyright (c) Facebook, Inc. and its affiliates.
import os
import sys
import numpy as np
import scipy.io as io
import timm
rng = np.random.RandomState(23456)
import math
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


class regressor_fcn_bn_32(nn.Module):
	def __init__(self):
		super(regressor_fcn_bn_32, self).__init__()

	def build_net(self, feature_in_dim, feature_out_dim, require_image=False, default_size=256):
		self.require_image = require_image
		self.default_size = default_size
		self.use_resnet = True
				
		embed_size = default_size

		# 如果有图片，则加入该部分网络，输入是resnet输出的特征向量 [2x512]
		if self.require_image:
			embed_size += default_size
			if self.use_resnet:
				#将[1024]映射到[256]
				self.image_resnet_postprocess = nn.Sequential(
					nn.Dropout(0.5),
					nn.Linear(512*2, default_size),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(default_size, momentum=0.01),
				)
				#nx64x256

				self.image_reduce = nn.Sequential(
					nn.MaxPool1d(kernel_size=2, stride=2),
				)

		# 处理手部序列输入
		self.encoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(feature_in_dim,256,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(256),
			nn.MaxPool1d(kernel_size=2, stride=2),
		)


		self.conv5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv6 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv7 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv8 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv9 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv10 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.skip1 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)
		self.skip2 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)
		self.skip4 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)
		self.skip5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.decoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),

			nn.Dropout(0.5),
			nn.ConvTranspose1d(embed_size, feature_out_dim, 7, stride=2, padding=3, output_padding=1),
			nn.ReLU(True),
			nn.BatchNorm1d(feature_out_dim),

			nn.Dropout(0.5),
			nn.Conv1d(feature_out_dim, feature_out_dim, 7, padding=3),
		)


	## create image embedding
	def process_image(self, image_):
		B, T, _ = image_.shape 
		image_ = image_.view(-1, 512*2)
		feat = self.image_resnet_postprocess(image_)
		feat = feat.view(B, T, self.default_size)
		feat = feat.permute(0, 2, 1).contiguous()
		feat = self.image_reduce(feat)
		return feat


	## utility upsampling function
	def upsample(self, tensor, shape):
		return tensor.repeat_interleave(2, dim=2)[:,:,:shape[2]] 


	## forward pass through generator
	def forward(self, input_, image_=None):
		B, T = input_.shape[0], input_.shape[2]

		fourth_block = self.encoder(input_)
		#nx256x32

		if self.require_image:
			feat = self.process_image(image_)
			#nx256x32
			fourth_block = torch.cat((fourth_block, feat), dim=1)
			#nx512x32
		
		fifth_block = self.conv5(fourth_block)
		#nx512x32

		sixth_block = self.conv6(fifth_block)
		#nx512x32

		seventh_block = self.conv7(sixth_block)
		#nx512x16

		eighth_block = self.conv8(seventh_block)
		#nx512x16

		ninth_block = self.conv9(eighth_block)
		#nx512x16

		tenth_block = self.conv10(ninth_block)
		#nx512x16

		ninth_block = tenth_block + ninth_block

		ninth_block = self.skip1(ninth_block)

		eighth_block = ninth_block + eighth_block
		eighth_block = self.skip2(eighth_block)

		sixth_block = self.upsample(seventh_block, sixth_block.shape) + sixth_block
		sixth_block = self.skip4(sixth_block)

		fifth_block = sixth_block + fifth_block
		fifth_block = self.skip5(fifth_block)

		output = self.decoder(fifth_block)
		return output 


class regressor_fcn_bn_discriminator(nn.Module):
	def __init__(self):
		super(regressor_fcn_bn_discriminator, self).__init__()

	def build_net(self, feature_in_dim):
		self.convs = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(feature_in_dim,64,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(64),
			## 64

			nn.Dropout(0.5),
			nn.Conv1d(64,64,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(64),
			## 32

			nn.Dropout(0.5),
			nn.Conv1d(64,32,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(32),
			## 16

			nn.Dropout(0.5),
			nn.Conv1d(32,32,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(32),
			## 8

			nn.Dropout(0.5),
			nn.Conv1d(32,16,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(16),
			## 4

			nn.Dropout(0.5),
			nn.Conv1d(16,16,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(16),
			## 2

			nn.Dropout(0.5),
			nn.Conv1d(16,8,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(8),
			## 1

			nn.Dropout(0.5),
			nn.Conv1d(8,1,3,padding=1),
		)

	def forward(self, input_):
		outputs = self.convs(input_)
		return outputs

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class body2handformer(nn.Module):
	def __init__(self):
		super(body2handformer, self).__init__()
	
	def build_net(self, feature_in_dim=6*6,  feature_out_dim=512, hand_dim = 252,
	       seq_length = 64,nhead = 8, dropout = 0.5,
		   num_encoder_layers =6,num_decoder_layers = 6, 
		   dim_feedforward = 2048,require_image=True,pos_embedding = PositionalEncoding(512,0.5,64),
		    mem_mask = False,output_mask = True ):
		self.require_image = require_image
		if mem_mask:
			self.mask_mem = Transformer.generate_square_subsequent_mask(seq_length).cuda()
		else:
			self.mask_mem = None
		if output_mask:
			self.mask_output = Transformer.generate_square_subsequent_mask(seq_length).cuda()
		else:
			self.mask_output = None

		# network begin token
		self.begin_token = nn.Parameter(torch.randn(1,1,hand_dim),requires_grad=True)
		# 定义 embedding ver 1.0
		if require_image:
			self.img_embbding = nn.Sequential(nn.Dropout(dropout),
					nn.Linear(1024, int(feature_out_dim/2)),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(seq_length, momentum=0.01))
			self.seq_embdding = nn.Sequential(	nn.Dropout(dropout),
					nn.Linear(feature_in_dim, int(feature_out_dim/2)),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(seq_length, momentum=0.01))
		else:
			self.seq_embdding = nn.Sequential(	nn.Dropout(dropout),
					nn.Linear(feature_in_dim, feature_out_dim),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(seq_length, momentum=0.01))
			
		self.hand_embdding = nn.Sequential(	nn.Dropout(dropout),
				nn.Linear(hand_dim, feature_out_dim),
				nn.LeakyReLU(0.2, True),
				nn.BatchNorm1d(seq_length, momentum=0.01))
			
		# 定义 pos embedding default cos sin
		self.pos_embdding = pos_embedding
		# 定义 transformer default 64 token 512 dim 
		self.transformer = Transformer(d_model=feature_out_dim,nhead = nhead,
				 num_encoder_layers = num_decoder_layers, num_decoder_layers = num_encoder_layers,
				 dim_feedforward = dim_feedforward,batch_first = True,dropout=dropout)
		# 定义 regression 
		# now there output 1/64 token 512 dim[nx1/64x512]
		# we should output 42*6 = 252
		self.out = nn.Sequential(nn.Dropout(dropout),
				nn.Linear(feature_out_dim, hand_dim))
				# nn.LeakyReLU(0.2, True),
				# nn.BatchNorm1d(feature_out_dim, momentum=0.01))
	def forward(self,seq_input,hand_input=None,img_input=None,inference=False):
		# seq
		x = self.seq_embdding(seq_input)
		# img
		if self.require_image:
			img_emd = self.img_embbding(img_input)
			# b x frame x feacture
			x = torch.concat((x,img_emd),dim = 2)
		# hand seq 64->63
		x = self.pos_embdding.forward(x)
		if not inference :
			hand_input_in = hand_input[:,:hand_input.shape[1]-1,:]
		else:
			hand_input_in = hand_input

		batch_size = seq_input.shape[0]
		begin_token = self.begin_token.expand(batch_size, -1, -1)
		hand_emb = torch.concat((begin_token,hand_input_in),dim=1)
		hand_emb = self.hand_embdding(hand_emb)
		# add begin token
		hand_emb = self.pos_embdding.forward(hand_emb)


		x = self.transformer.forward(x,hand_emb,memory_mask=self.mask_mem,tgt_mask=self.mask_output)
		return x
	
	# def train(self,seq_input,hand_input,img_input=None):
	# 	# 64*512
	# 	x = self.forward(self,seq_input,hand_input,img_input)
	# 	x = self.out(x)
	# 	return x
	

	# 也使用transformer架构，判断是否是真实的手型序列、只使用编码器
class body2handformer_discriminator(nn.Module):
	def __init__(self):
		super(body2handformer_discriminator, self).__init__()
	def build_net(self, feature_out_dim=512, hand_dim = 252,
	       seq_length = 64,nhead = 8, dropout = 0.1,
		   num_encoder_layers =6, 
		   feedforward_dim = 2048,pos_embedding = PositionalEncoding(512,0.1,64+1)):
		self.cls_token  = nn.Parameter(torch.randn(1,1,hand_dim),requires_grad=True)
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_out_dim, nhead=nhead,
						    dim_feedforward=feedforward_dim,dropout=dropout)
		self.encoder = nn.TransformerEncoder(self.encoder_layer,num_decoder_layers=num_encoder_layers)
		self.pos_emb = pos_embedding
		self.project = nn.Sequential(nn.Dropout(dropout),
					nn.Linear(hand_dim, feature_out_dim),
					nn.LeakyReLU(0.2, True),
					nn.BatchNorm1d(seq_length, momentum=0.01)
		)
		# mse or bceloss
		self.head = nn.Sequential(nn.Dropout(dropout),
					nn.Linear(feature_out_dim, 1)
		)
		
	def forward(self,xinput):
		cls_token = self.cls_token.expand(xinput.shape[0],-1,-1)
		xinput = self.project(input)
		x = torch.cat((cls_token,xinput))
		x = self.pos_emb.forward(x)
		x = self.encoder.forward(x)
		# 只对cls token作梯度下降
		x = self.head(x[:,0,:])
		return x

#
# class body2handformer_discriminator(nn.Module):
# 	def __init__(self):
# 		super(body2handformer_discriminator, self).__init__()
# 	def build_net(self,):



