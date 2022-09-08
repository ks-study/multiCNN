import torch
import random
import torchvision.transforms as transforms
import numpy as np



class ElimMargin:
    def __init__(self,margin=20):
        self.margin_elim=margin
        print("load em")
    def __call__(self,volume):
        margin_elim=self.margin_elim
#         print(volume.shape)
        new_volume=volume[:,margin_elim:-margin_elim,margin_elim:-margin_elim]
#         print(new_volume.shape)
        return new_volume

def standalize(x,means,stds,i_ch=None):
    if i_ch !=None:
        means=[means[i_ch]]
        stds=[stds[i_ch]]
        # print("before xmax",x.max(),"xmin",x.min())
        x = x.sub(torch.FloatTensor(means).view(1, 1))
        x = x.div(torch.FloatTensor(stds).view(1, 1))
        # x = x.mul(torch.FloatTensor(stds).view( 1, 1))
        # x = x.add(torch.FloatTensor(means).view( 1, 1))
        # print("after xmax",x.max(),"xmin",x.min())
        # print(x)
        # print("min_",np.min(x.numpy()))
        # print("max_",np.max(x.numpy()))
        return x
    else:
        x = x.sub(torch.FloatTensor(means).view(volume.shape[0], 1, 1))
        x = x.div(torch.FloatTensor(stds).view(volume.shape[0], 1, 1))
    return x
def normalize(volume):
    """Normalize the volume"""
    min = -200.0
    max = 400.0
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume
def unchange(volume):
    return volume
class NormalizeRange(object):
    def __init__(self,ch_num=3,range_list=None):
        if range_list==None:
            range_list=[[-200,400]]*ch_num
        self.range_list=range_list
    def __call__(self,volume):
        # print(volume.shape)
        for i_ch in range(volume.shape[-1]):
            _range=self.range_list[i_ch]
            
            volume_ch=volume[:,:,i_ch]
            volume_ch[volume_ch < _range[0]] = _range[0]
            volume_ch[volume_ch > _range[1]] = _range[1]
            # volume_ch[:,:] = (volume_ch - _range[0])
            volume_ch[:,:] = (volume_ch - _range[0]) / (_range[1] - _range[0])
        # print("norm min",volume.min())
        # print("norm max",volume.max())
        # print(volume[:4,:5,:1])
        volume = volume.astype("float32")
        return volume


class ColorShift(object):
    def __init__(self,delay_range=[0,10],p=0.3):
        self.delay_range=delay_range
        self.p=p
    def __call__(self,volume):
        rd=random.uniform(0,1)
        if rd>=self.p:
            return volume
        delays=[]
        for i in range(4):
            delays.append(random.randint(self.delay_range[0],self.delay_range[1]))
    #         delay2=random.randint(self.delay_range[0],self.delay_range[1])
        imgs=[volume[i] for i in range(3)]
        imgs[0]=torch.roll(imgs[0], (delays[0],delays[1]),dims=(0,1))
        imgs[2]=torch.roll(imgs[2], (delays[2],delays[3]),dims=(0,1))
        volume[0]= imgs[0]
        volume[2]= imgs[2]
    #     volume[0]
        return volume

def as_float(volume):
    max_x=500
    min_x=-200
    volume[volume < min_x] = min_x
    volume[volume > max_x] =max_x
    volume = (volume - min_x) / (max_x - min_x)
    return volume.astype("float32")
    
def data_transform_train_def(img_size,ch_num,normalize_range_list=None,mask=False,use_normalize=True,means=None,stds=None):
    if means!=None and stds!=None:
        pass
    elif ch_num<=3:
        means=[0.485, 0.456, 0.406]
        stds=[0.229, 0.224, 0.225]
    else:
        means=[0.485, 0.456, 0.406]+[0.485]*(ch_num-3)
        stds=[0.229, 0.224, 0.225]+[0.229]*(ch_num-3)
    tf_list=[
        # ,
        # normalize,
        transforms.ToTensor(),
        
        ColorShift([0,0]),
#         ElimMargin(28),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.5), ratio=(0.95,1.05), interpolation=2),
        
        transforms.RandomRotation(degrees=40),
        
        
        transforms.RandomHorizontalFlip(p=0.5),

        
        # transforms.Normalize(means,stds),
    ]
    # if use_normalize:
    #     tf_list.append(transforms.Normalize(means,stds))
    tf_main=transforms.Compose(tf_list)
    tf_pre=NormalizeRange(ch_num,normalize_range_list)
    tf_post_list=[transforms.RandomErasing(p=0.3, scale=(0.01, 0.03), ratio=(0.5, 1.5), value=0)]
    if use_normalize:
        tf_post_list.append(transforms.Normalize(means,stds))
    tf_post=transforms.Compose(tf_post_list)
    return tf_pre,tf_main,tf_post





def data_transform_unet_train_def(img_size,ch_num=None,normalize_range_list=None,use_normalize=True,means=None,stds=None):
    if means!=None and stds!=None:
        pass
    elif ch_num<=3:
        means=[0.485, 0.456, 0.406]
        stds=[0.229, 0.224, 0.225]
    else:
        means=[0.485, 0.456, 0.406]+[0.406]*(ch_num-3)
        stds=[0.229, 0.224, 0.225]+[0.225]*(ch_num-3)
    tf_list=[
        NormalizeRange(ch_num,normalize_range_list),
        transforms.ToTensor(),
        # ColorShift([0,0]),
#         ElimMargin(28),
        transforms.RandomResizedCrop(img_size, scale=(0.4, 1.5), ratio=(0.95,1.05), interpolation=2),
        
        # transforms.RandomRotation(degrees=40),
        
        
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomErasing(p=0.3, scale=(0.01, 0.03), ratio=(0.5, 1.5), value=0),
    ]
    if use_normalize:
        tf_list.append(transforms.Normalize(means,stds))
    return transforms.Compose(tf_list)
def un_normalize(x,ch_num,i_ch,means=None,stds=None):
    if means!=None and stds!=None:
        pass
    elif ch_num<=3:
        means=[0.485, 0.456, 0.406]
        stds=[0.229, 0.224, 0.225]
    else:
        means=[0.485, 0.456, 0.406]+[0.406]*(ch_num-3)
        stds=[0.229, 0.224, 0.225]+[0.225]*(ch_num-3)
    if i_ch !=None:
        means=[means[i_ch]]
        stds=[stds[i_ch]]
        # print("before xmax",x.max(),"xmin",x.min())
        x = x.mul(torch.FloatTensor(stds).view( 1, 1))
        x = x.add(torch.FloatTensor(means).view( 1, 1))
        # print("after xmax",x.max(),"xmin",x.min())
        # print(x)
        # print("min_",np.min(x.numpy()))
        # print("max_",np.max(x.numpy()))
        return x
    else:
        x = x.mul(torch.FloatTensor(stds).view(ch_num, 1, 1))
        x = x.add(torch.FloatTensor(means).view(ch_num, 1, 1))
        return x
def data_transform_unet_test_def(img_size,ch_num,normalize_range_list=None,use_normalize=True,means=None,stds=None):
    if means!=None and stds!=None:
        pass
    elif ch_num<=3:
        means=[0.485, 0.456, 0.406]
        stds=[0.229, 0.224, 0.225]
    else:
        means=[0.485, 0.456, 0.406]+[0.406]*(ch_num-3)
        stds=[0.229, 0.224, 0.225]+[0.225]*(ch_num-3)
    tf_list=[
        NormalizeRange(ch_num,normalize_range_list),
        transforms.ToTensor(),
        # ColorShift([0,0]),
#         ElimMargin(28),
        transforms.RandomResizedCrop(img_size, scale=(0.99, 1.5), ratio=(0.95,1.05), interpolation=2),
        # transforms.RandomRotation(degrees=40),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomErasing(p=0.3, scale=(0.01, 0.03), ratio=(0.5, 1.5), value=0),
        # transforms.Normalize(means,stds),
    ]
    if use_normalize:
        tf_list.append(transforms.Normalize(means,stds))
    return transforms.Compose(tf_list)


def data_transform_test_def(img_size,ch_num,normalize_range_list=None,use_normalize=True,means=None,stds=None):
    if means!=None and stds!=None:
        pass
    elif ch_num<=3:
        means=[0.485, 0.456, 0.406]
        stds=[0.229, 0.224, 0.225]
    else:
        means=[0.485, 0.456, 0.406]+[0.406]*(ch_num-3)
        stds=[0.229, 0.224, 0.225]+[0.225]*(ch_num-3)
    tf_pre=NormalizeRange(ch_num,normalize_range_list)
    tf_post=transforms.Normalize(means,stds)
    tf_list=[
        # normalize,
        # NormalizeRange(ch_num,normalize_range_list),
        transforms.ToTensor(),
        
        ColorShift([0,0]),
#         ElimMargin(28),
        transforms.RandomResizedCrop(img_size, scale=(0.99, 1.5), ratio=(0.95,1.05), interpolation=2),
          
    ]
    # if use_normalize:
    #     tf_list.append(transforms.Normalize(means,stds))
    tf_main=transforms.Compose(tf_list)
    tf_post_list=[unchange]
    if use_normalize:
        tf_post_list.append(transforms.Normalize(means,stds))
    tf_post=transforms.Compose(tf_post_list)
    return tf_pre,tf_main,tf_post