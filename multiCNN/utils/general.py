import pandas as pd
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from .transforms import normalize
from .transforms import un_normalize
from .transforms import standalize
import re
from torch.nn.parameter import Parameter
def dic_add(dic_a,dic_b,depth=1):
    dic_res={}
    dic_a_keys=list(dic_a.keys())
    dic_ab_keys_sorted=sorted(dic_a.keys()&dic_b.keys(),key=dic_a_keys.index)
    if depth>=2:
        for k in dic_ab_keys_sorted:
            dic_res[k]=dic_add(dic_a[k],dic_b[k],depth=depth-1)
    else:
        for k in (dic_ab_keys_sorted):
            res_elm = [val_a + val_b for val_a,val_b in zip(dic_a[k],dic_b[k])]
            dic_res[k]=res_elm
    return dic_res
def dic_div(dic_a,division,depth=1):
    dic_res={}
    if depth>=2:
        for k in (dic_a.keys()):
            dic_res[k]=dic_div(dic_a[k],division,depth=depth-1)
    else:
        for k in dic_a.keys():
            dic_res[k]=[val/division for val in dic_a[k]]
    return dic_res
def dic_minus(dic_a,dic_b,depth=1):
    dic_res={}
    if depth>=2:
        for k in (dic_a.keys()):
            dic_res[k]=dic_minus(dic_a[k],dic_b[k],depth=depth-1)
    else:
        for k in (dic_a.keys() & dic_b.keys()):
            res_elm = [val_a - val_b for val_a,val_b in zip(dic_a[k],dic_b[k])]
            dic_res[k]=res_elm
    return dic_res

def dic_power(dic_a,power,depth=1):
    dic_res={}
    if depth>=2:
        for k in (dic_a.keys()):
            dic_res[k]=dic_power(dic_a[k],power,depth=depth-1)
    else:
        for k in dic_a.keys():
            dic_res[k]=[val**power for val in dic_a[k]]
    return dic_res
    
def show_accession_bbs(accession_id,foot_tag="_registered"):
    df=pd.read_csv('/data1/RCC/accession_DB_merged.csv',index_col=0)
    _type=df.at[int(accession_id),"diagnosis"]
    for k in range(1):
        img_arr_list=[]
        for layer_id in range(4):
            path=f"/data1/RCC/shono_dicom2/dicom/{accession_id}/{layer_id}-bb-Original-1.25{foot_tag}.nii.gz"
            if not os.path.exists(path):
                print("error")
                return 0
            sitk_img=sitk.ReadImage(path)
            img_arr_list.append(sitk.GetArrayFromImage(sitk_img))
        fig,axs=plt.subplots(1,4,figsize=(16,4))
        fig2,axs2=plt.subplots(1,4,figsize=(16,4))
        for layer_id,img_arr in enumerate(img_arr_list):
            img_arr=normalize(img_arr)
            axs[layer_id].imshow(img_arr[img_arr.shape[0]//2],cmap="bone",vmin=0.0,vmax=1.0)
            axs2[layer_id].imshow(img_arr[:,:,img_arr.shape[1]//2],cmap="bone",vmin=0.0,vmax=1.0)
        plt.show()

pattern_end=re.compile("^[\s\S]*[^\d](\d+)$")
def get_last_number(str):
    m=pattern_end.match(str)
    return int(m.group(1))

def show_image_phases(image,save_path=None,means=None,stds=None,vmin=None,vmax=None,cmap=None,use_norm=False):
    if cmap==None:
        cmap="bone"
    # fig = plt.figure()
    # shapeが[ch_n,H,W]のtensorをチャンネルごとに表示する。
    ch_num=image.shape[0]
    fig,axs=plt.subplots(1,ch_num,figsize=(ch_num*4,4))
    print(ch_num)
    for i_ch in range(ch_num):
        img_arr=image[i_ch].cpu().detach()
        # img_arr_no_norm=un_normalize(img_arr,ch_num,i_ch,means,stds)
        img_arr_no_norm=img_arr
        if means!=None:
            if use_norm:
                img_arr_no_norm=standalize(img_arr,means,stds,i_ch)
            else:
                img_arr_no_norm=un_normalize(img_arr,ch_num,i_ch,means,stds)
        print(np.min(img_arr_no_norm.numpy()))
        print(np.max(img_arr_no_norm.numpy()))
        # print(img_arr_no_norm.max(),img_arr_no_norm.min())
        if means!=None:
            if vmin!=None:
                axs[i_ch].imshow(img_arr_no_norm,cmap=cmap,vmin=vmin,vmax=vmax)
            else:
                axs[i_ch].imshow(img_arr_no_norm,cmap=cmap)
        else:
            if vmin!=None:
                axs[i_ch].imshow(img_arr_no_norm,cmap=cmap,vmin=vmin,vmax=vmax)
            else:
                axs[i_ch].imshow(img_arr_no_norm,cmap=cmap,vmin = -0.7, vmax = 0.7)
        # axs[i_ch].imshow(img_arr_no_norm,cmap="bone",vmin = -15, vmax = 15)
    plt.show()
    if save_path!=None:
        fig.savefig(save_path)

import pickle
import io
import torch

class PickleCPU(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_backbone_state_dict(model, state_dict):
        own_state = model.state_dict()
        unloaded_state=0
        for name, param in state_dict.items():
            if name[:8]=="backbone":
                name=name[9:]
            if name not in own_state or "fc" in name:
                 continue
            unloaded_state+=1
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
        print("unloaded",unloaded_state,"layers")
