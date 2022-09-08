import random
import pandas as pd
import os
import re
import pickle
from torch.utils.data import Dataset
import numpy as np
import glob
from collections import Counter
import  matplotlib.pyplot  as plt
import copy as cp
import torch

from multiCNN.utils.general import get_last_number

import SimpleITK as sitk

filters = {
    "AdditiveGaussianNoise" : sitk.AdditiveGaussianNoiseImageFilter(),
    "Bilateral" : sitk.BilateralImageFilter(),
    "BinomialBlur" : sitk.BinomialBlurImageFilter(),
    "BoxMean" : sitk.BoxMeanImageFilter(),
    "BoxSigmaImageFilter" : sitk.BoxSigmaImageFilter(),
    "CurvatureFlow" : sitk.CurvatureFlowImageFilter(),
    "DiscreteGaussian" : sitk.DiscreteGaussianImageFilter(),
    "LaplacianSharpening" : sitk.LaplacianSharpeningImageFilter(),
    "Mean" : sitk.MeanImageFilter(),
    "Median" : sitk.MedianImageFilter(),
    "Normalize" : sitk.NormalizeImageFilter(),
    "RecursiveGaussian" : sitk.RecursiveGaussianImageFilter(),
    "ShotNoise" : sitk.ShotNoiseImageFilter(),
    "SmoothingRecursiveGaussian" : sitk.SmoothingRecursiveGaussianImageFilter(),
    #"SpeckleNoise" : sitk.SpeckleNoiseImageFilter(),
}


def generate_shuffled_list(list_path="/data1/RCC/shono_dicom2/useful_accessions.txt"):
    df=pd.read_csv('/data1/RCC/accession_DB_merged.csv',index_col=0)
    accession_id_list=[str(accession_id) for accession_id in df.index.values]
#     with open(list_path,"r") as f:
#             txt=f.read()
#     accession_id_list=txt.split(",")
    random.shuffle(accession_id_list)
    txt=",".join(accession_id_list)
    save_name=os.path.dirname(list_path)+"/useful_accessions_shuffled.txt"
    with open(save_name,"w") as f:
        f.write(txt)
        


def get_label(accession_id,label):
    df=pd.read_csv('/data1/RCC/RCC_detail_done.csv' ,encoding="shift-jis",index_col="accession")
    
    return df.at[int(accession_id),label]

def split_array(ar, n_group):
    for i_chunk in range(n_group):
        yield ar[i_chunk * len(ar) // n_group:(i_chunk + 1) * len(ar) // n_group]
        
def dic_min_len(dic):
    len_dic={}
    for key,val in dic.items():
        len_dic[key]=len(val)
    max_key = min(len_dic, key=len_dic.get)
    return max_key
#EFFECTS: テストデータセットと訓練データセットに対応するaccessionナンバーのリストを作成する。
#この時、欠損データを無視する。
#この時、不使用データ（diagnosisが不適切など）を除去する。
def Create_dataset_txt(
    list_path="/data1/RCC/shono_dicom2/useful_accessions.txt",
    save_dir="/data1/RCC/shono_dicom2/",grouping="diagnosis",
    feature_partition=None,
    division=4,test_chunk=0,shuffle=False):
    df=pd.read_csv('/data1/RCC/accession_DB_merged.csv',index_col=0)
    df=df.query("source=='Keio'")
    df2=pd.read_csv('/data1/RCC/RCC_detail_done.csv' ,encoding="shift-jis",index_col="accession")
    df2=df2.dropna(subset=['画質'])
    accession_id_list=sorted(list(set(df.index.values) & set(df2.index.values)))
    accession_id_list=[str(accession_id) for accession_id in accession_id_list]
#     accession_id_list=[str(accession_id) for accession_id in df.index.values]
#     accession_id_list=df.index.values
    if shuffle:
        random.shuffle(accession_id_list)
#     df=pd.read_csv('/data1/RCC/RCC_total.csv',index_col=0)
    types_features=["0","1","2"]
    if feature_partition==None:
        feature_partition={grouping:3}
    
    if grouping=="diagnosis":
        types_diagnosis=["clear","chromophobe","papillary"]
        types_grouping=types_diagnosis
    else:
        types_features=list(range(feature_partition[grouping]))
        types_features=[str(_type) for _type in types_features]
        types_grouping=types_features

    RCC_type_accession_id_list={}
    for _type in types_grouping:
        RCC_type_accession_id_list[_type]=[]
    # _type_list=[]
    for accession_id in accession_id_list:
        if grouping=="diagnosis":
            _type=df.at[int(accession_id),"diagnosis"]
        else:
            feature_denominator=9//feature_partition[grouping]
            _type=str((int(get_label(accession_id,grouping))-1)//feature_denominator)
        # _type_list.append(_type)
        RCC_type_accession_id_list[_type].append(accession_id)
    RCC_type_testable_id_list={}
    RCC_type_untestable_id_list={}
    min_len_type=dic_min_len(RCC_type_accession_id_list)
    min_len=150
    min_len_list=[min_len]*len(types_grouping)
    print("min_len_type:",min_len_type)
    print("min_len:",min_len)
    for type_i, _type in enumerate(types_grouping):
        RCC_type_testable_id_list[_type]=(RCC_type_accession_id_list[_type][:min_len_list[type_i]])
        RCC_type_untestable_id_list[_type]=RCC_type_accession_id_list[_type][min_len_list[type_i]:]
        print(f"type_list_{_type} len: ",len(RCC_type_testable_id_list[_type]))
    print(f"len:{len(accession_id_list)}")
    train_id_list=[]
    test_id_list=[]
    RCC_accession_group_list={}
    for i,_type in enumerate(types_grouping):
        print(f"{_type} num:{len(RCC_type_testable_id_list[_type])}")
        RCC_accession_group_list[_type]=list(split_array(RCC_type_testable_id_list[_type],division))
        print(f"div_len {_type} :",len(RCC_accession_group_list[_type]))
        chunk_id=test_chunk+i
        while chunk_id>=len(RCC_accession_group_list[_type]):
            chunk_id-=len(RCC_accession_group_list[_type])
        print("chunk_id:",chunk_id)
        test_id_list+=RCC_accession_group_list[_type].pop(chunk_id)
        for accession_group_idx in range(len(RCC_accession_group_list[_type])):
            train_id_list+=RCC_accession_group_list[_type][accession_group_idx]
#         train_id_list+=RCC_type_untestable_id_list[_type]
    
#訓練データとテストデータを分割

    print(f"train_len:{len(train_id_list)}")
    print(f"test_len:{len(test_id_list)}")
    # count_features={feature_value:0 for feature_value in types_features}
    # for accession_id in test_id_list:
    #     feature=int(get_label(accession_id,"早期濃染"))
    #     feature=str((feature-1)//3)
    #     count_features[feature]+=1
    _type_list=[]
    for ac_id in test_id_list:
        _type=df.at[int(ac_id),"diagnosis"]
        _type_list.append(_type)
    print("type_count",Counter(_type_list))
    
    with open(save_dir+'/train_multi_features.data', 'wb') as f:
        pickle.dump(train_id_list,f)
    with open(save_dir+'/test_multi_features.data', 'wb') as f:
        pickle.dump(test_id_list,f)
    
def get_numpy_slice_id(slice_path):
    name=os.path.basename(slice_path)
    slice_id=re.sub(r"([^\-]+-[^\-]+-)?([0-9]+)(_no_pad)?.npy",r"\2",name)
    return int(slice_id)

def get_path_format_from_accession_id(accession_id):
    df=pd.read_csv('/data1/RCC/accession_DB_merged.csv',index_col=0)
    source=df.at[int(accession_id),"source"]
    path=""
    if source=="Keio":
        path=f"/data1/RCC/shono_dicom2/npy_2ds/{accession_id}/Original-1.25-*[0-9]_no_pad.npy"
#         path=f"/data1/RCC/shono_dicom2/npy_2ds_without_elim/{accession_id}/Original-1.25-*[0-9]_no_pad.npy"
    if source=="JMID":
        path=f"/data1/RCC/shono_dicom2/npy_2ds/{accession_id}/Original-1.25-*[0-9]_no_pad.npy"
#         path=f"/data1/RCC/shono_dicom2/npy_2ds_without_elim/{accession_id}/[0-9].npy"
    return path


#ABSTRACT : Pytorchのデータセットを作成するクラス。
#ここでは4層を持つ腫瘍画像を4チャンネル画像として入力する。

class Create_Dataset(Dataset):
    def __init__(self,dataset_path,root_path="/data1/RCC/shono_dicom2/npy_2ds",
    data_transform=None,spacing_xy=None,spacing_z=1.25,ch_list=[0,1,2],
    z_shift_range=[0,0],slice_section=5,feature_partition=None,new_diagnosis_cor=[0,1,2],
    loader_out_task_names=["早期濃染"],radiologic_feature_names=["早期濃染"],crop_by_mask=None,use_bce=False,bce_noise_std=None):
        self.dataset_path=dataset_path
        self.df=pd.read_csv('/data1/RCC/accession_DB_merged.csv',index_col=0)
        self.root_path=root_path
        if type(dataset_path)!=list:
            with open(dataset_path, 'rb') as f:
                accession_list=pickle.load(f)
        self.accession_path_list=[f"{root_path}/{accession}" for accession in accession_list]
#         random.shuffle(self.accession_path_list)
        self.data_transform=data_transform
        self.spacing_xy=spacing_xy
        self.spacing_xy_name=spacing_xy
        if self.spacing_xy_name==None:
            self.spacing_xy_name="Original"
        self.spacing_z=spacing_z
        self.ch_list=ch_list
        self.z_shift_range=z_shift_range
        self.slice_section=slice_section
        self.radiologic_feature_names=radiologic_feature_names

        self.loader_out_task_names=loader_out_task_names
        if feature_partition==None:
            feature_partition={feature_name:3 for feature_name in radiologic_feature_names}
        self.feature_partition=feature_partition
        self.new_diagnosis_cor=new_diagnosis_cor
        self.crop_by_mask=crop_by_mask
        # self.filter=None
        # self.filter=filters["LaplacianSharpening"]
        self.filter=None
        self.use_bce=use_bce
        self.bce_noise_std=bce_noise_std
    def set_data_transform(self,data_transform):
        self.data_transform=data_transform
    def __len__(self):
        return len(self.accession_path_list)
    def get_item_by_accession_id(self,accession_id):
        slice_path_format=get_path_format_from_accession_id(accession_id)
        slice_list=sorted(glob.glob(slice_path_format),key=get_numpy_slice_id)
        len_slice_list=len(slice_list)
        slice_section_min=self.slice_section//2
        slice_section_max=self.slice_section//2+1
        slice_id=random.randint(len_slice_list*slice_section_min//self.slice_section,len_slice_list*slice_section_max//self.slice_section)
        ch_list=[item+1 for item in self.ch_list]
        slice_id_list=[]
        z_shift_range=self.z_shift_range
        for ch_num in ch_list:
            if ch_num==3: #2+1
                slice_id_list.append(slice_id)
            else:
                shift_dist=random.randint(z_shift_range[0],z_shift_range[1])
                slice_id_shifted=max(0,min(slice_id+shift_dist,len(slice_list)-1))
                slice_id_list.append(slice_id_shifted)
        images=[]
        masks=[]
        for i,(slice_id_ch,ch_num) in enumerate(zip(slice_id_list,ch_list)):
            data_path=slice_list[slice_id_ch]
            data=np.load(data_path,allow_pickle=True)
            diagnosis=data[0]
            mask_=data[5]
            masks.append(mask_)
            # masked_by_tumor=True
            image_phase=data[ch_num]
            # plt.imshow(image_phase)
            # plt.show()
            if self.crop_by_mask!=None:
                mask_crop=cp.copy(mask_)
                if type(self.crop_by_mask)==list:
                    crop_by_mask_ratio=self.crop_by_mask[i]
                else:
                    crop_by_mask_ratio=self.crop_by_mask
                mask_crop[mask_crop==0]=crop_by_mask_ratio
                image_phase=image_phase*(mask_crop+crop_by_mask_ratio)
                # masks.append()
            # print("data_lennnn",len(data))
            images.append(image_phase)
            # plt.imshow(image_phase)
            # plt.show()
        
        # print(mask_.shape)
        # print(mask_)
        # print(np.max(mask_))
        # plt.imshow(mask_)
        # plt.show()
        image_3d=np.stack(list(images),-1)
        mask_3d=np.stack(list(masks),-1)
        # print("img_max",image_3d.max())
        # print("img_min",image_3d.min())
        # plt.imshow(mask_3d)
        # plt.show()
        if self.filter!=None:
            image_3d=sitk.GetImageFromArray(np.array(image_3d))
            image_3d=sitk.GetArrayFromImage(self.filter.Execute(image_3d))
        
        if "mask" in self.loader_out_task_names:
            image=self.data_transform[0](image_3d)
            
            # print("image_shape",image.shape)
            image_and_mask=np.concatenate([image,mask_3d],2)
            # print("image_and_mask_shape",image_and_mask.shape)
            image_and_mask=self.data_transform[1](image_and_mask)
            image,mask=torch.tensor_split(image_and_mask,2,dim=0)
            image=self.data_transform[2](image)
        else:
            # print("img_get 0 max",image_3d[0].max(),"max",image_3d[0].min())
            # print("img_get 1 max",image_3d[1].max(),"max",image_3d[1].min())
            # print("img_get 2 max",image_3d[2].max(),"max",image_3d[2].min())
            image=self.data_transform[0](image_3d)
            image=self.data_transform[1](image)
            image=self.data_transform[2](image)
            # print("after img_get 0 max",image_3d[0].max(),"max",image_3d[0].min())
            # print("after img_get 1 max",image_3d[1].max(),"max",image_3d[1].min())
            # print("after img_get 2 max",image_3d[2].max(),"max",image_3d[2].min())
        # plt.imshow(mask)
        # plt.show()
        # print("img2_max",image.max())
        # print("img2_min",image.min())
        
        # np.stack(list(images),-1)
        label_data={}
        for task_name in self.loader_out_task_names:
            if task_name=="diagnosis":
                label_data[task_name]=self.new_diagnosis_cor[diagnosis]
            elif "diagnosis_bin" in task_name:
                diagnosis=self.new_diagnosis_cor[diagnosis]
                i_bin=get_last_number(task_name)
                label_data[task_name]=int(i_bin==diagnosis)
            elif task_name=="mask":
                label_data[task_name]=mask
            else:
                feature=get_label(accession_id,task_name)
                if self.use_bce:
                    feature=(feature-1)/(9-1)
                    if self.bce_noise_std!=None:
                        noise=random.gauss(0,self.bce_noise_std)
                        feature=feature+noise
                        if feature>1:
                            feature=1.0
                        if feature<0:
                            feature=0.0
                    # print(feature)
                else:
                    feature=int(feature)
                    feature=(feature-1)//(9//self.feature_partition[task_name])
                label_data[task_name]=feature
        # new_diagnosis_cor=[0,1,2]
        #[clear,chomophobe,papillary]の順番
        
        return label_data,image,accession_id
    def __getitem__(self,i):
        accession_path=self.accession_path_list[i]
        accession_id=os.path.basename(accession_path)
        return self.get_item_by_accession_id(accession_id)

        