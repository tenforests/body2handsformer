import os
from torch.utils.data import Dataset
import numpy as np
# from transforms import *

class b2hDataset(Dataset):
    def __init__(self,frame:64,npy_path,img_need:False,ratio:0.7,dataPaths:None,args):
        self.frame = frame
        self.npy_path = npy_path
        self.args = args
        self.array_hand,self.array_body,self.array_img,self.train_idx,self.test_idx = self.loadData(frame,npy_path,img_need,dataPaths,ratio,args)
        self.ARMS_ONLY = [12,13,14,15,16,17]
        self.FEATURE_MAP = {
    'arm2wh':((6*6), 42*6),
                }

    def loadData(self,frame,img:False,dataPaths,ratio,args):
        totalfiles = np.array()
        totalimgs = np.array()
        totalhands = np.array()
        totalbodys = np.array()
        for path in dataPaths:
            path = os.path.join(args.base_path,path)
            file_path = os.path.join(path, 'filepaths.npy')
            hand_path = os.path.join(path, 'full_hands2.npy')
            img_path = os.path.join(path,'full_resnet.npy')
            body_path = os.path.join(path,'full_bodies2.npy')
            if os.path.exists(file_path):
                files = np.load(file_path, allow_pickle=True)
                totalfiles = np.concatenate((totalfiles,files))
            if os.path.exists(hand_path):
                hands = np.load(hand_path, allow_pickle=True)
                totalhands = np.concatenate((totalhands,hands))
            if os.path.exists(img_path) and img:
                imgs = np.load(img_path, allow_pickle=True)
                totalimgs = np.concatenate((totalimgs,imgs))
            if os.path.exists(body_path):
                bodys = np.load(body_path, allow_pickle=True)
                bodys = np.reshape(bodys,(bodys.shape[0],bodys.shape[1],-1,6))
                bodys = bodys[:,:,self.ARMS_ONLY,:]
                bodys = np.reshape(bodys,(bodys.shape[0],bodys.shape[1],-1))
                totalbodys = np.concatenate((totalbodys,bodys))
        if frame !=64:
            totalhands,totalbodys,totalimgs = self.frameCaluate(files,hands,bodys,frame,imgs)
        total_num = totalhands.shape[0]
        train_N = int(total_num*ratio)
        idx = np.random.permutation(total_num)
        train_idx, test_idx = idx[:train_N], idx[train_N:]
        body_mean_X, body_std_X, body_mean_Y, body_std_Y = self.calc_standard(totalbodys[train_idx,:,:], totalhands[train_idx,:,:])
        np.savez_compressed(args.model_path + '{}{}_preprocess_core.npz'.format(args.tag, args.pipeline), 
            body_mean_X=body_mean_X, body_std_X=body_std_X,
            body_mean_Y=body_mean_Y, body_std_Y=body_std_Y) 
        totalbodys = (totalhands - body_mean_X)/body_std_X
        totalhands = (totalbodys - body_mean_Y)/body_std_Y
        return totalhands,totalbodys,totalimgs,train_idx,test_idx

    def calc_standard(self,train_X, train_Y):
        EPSILON = 1e-10
        body_mean_X = train_X.mean(axis=1).mean(axis=0)[np.newaxis, np.newaxis, :]
        body_std_X = train_X.std(axis=1).std(axis=0)[np.newaxis,np.newaxis, :]
        body_std_X += EPSILON
        body_mean_Y = train_Y.mean(axis=1).mean(axis=0)[np.newaxis, np.newaxis,:]
        body_std_Y = np.array([[[train_Y.std()]]]).repeat(train_Y.shape[2], axis=2)
        return body_mean_X, body_std_X, body_mean_Y, body_std_Y
    
    # def mean_std(self,feat, data):
        # EPSILON = 1e-10
        # if feat == 'wh':
        #     mean = data.mean(axis=1).mean(axis=0)[np.newaxis, np.newaxis, :]
        #     std =  data.std(axis=1).std(axis=0)[np.newaxis,np.newaxis, :]
        #     std += EPSILON
        # else:
        #     mean = data.mean(axis=1).mean(axis=0)[np.newaxis, np.newaxis,:]
        #     std = np.array([[[data.std()]]]).repeat(data.shape[1], axis=1)
        # return mean, std
    
    def frameCaculate(array_file,array_hand,array_body,frame_num:32,img_array:None):
        old_num = 64
        if old_num % frame_num != 0 :
            raise ValueError("frame_num为64的倍数")
        
        div = old_num/frame_num 
        array_file = array_file[::2,:,:]
        array_file = np.reshape(array_file,(array_file.shape[0]*div*2,frame_num/2,-1))
        array_hand = array_hand[::2,:,:]
        array_hand = np.reshape(array_hand,(array_hand.shape[0]*div*2,frame_num/2,-1) )
        array_body = array_body[::2,:,:]
        array_body = np.reshape(array_body,(array_body.shape[0]*div*2,frame_num/2,-1))
        if img_array != None:
            img_array = img_array[::2,:,:]
            img_array = np.reshape(img_array,(array_body.shape[0]*div*2,frame_num/2,-1))
        i = 1
        new_files = np.array()
        new_hands = np.array()
        new_bodys = np.array()
        new_imgs = np.array()
        while i < array_file.shape[0]:
            if array_file[i,0,0]==array_file[i-1,0,0]:
                # new_file = np.concatenate((array_file[i,0,0],array_file[i-1,0,0]),axis=1)
                # new_files = np.concatenate((new_files,new_file))
                new_hand = np.concatenate((array_hand[i,0,0],array_hand[i-1,0,0]),axis=1)
                new_hands = np.concatenate((new_hands,new_hand))
                new_body = np.concatenate((array_body[i,0,0],array_body[i-1,0,0]),axis=1)
                new_bodys = np.concatenate((new_bodys,new_body))
                if img_array != None:
                    new_img = np.concatenate((img_array[i,0,0],img_array[i-1,0,0]),axis=1)
                    new_imgs = np.concatenate((new_imgs,new_img))
        return new_hands,new_bodys,new_imgs

    def __getitem__(self,index):
        