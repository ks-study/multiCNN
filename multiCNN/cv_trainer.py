import copy as cp
# from typing import NewType
import torch
import pandas as pd
import os
import re


import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import sklearn.metrics
import pickle

import multiCNN.trainer as trainer
from multiCNN.data_management import Create_Dataset
from multiCNN.data_management import Create_dataset_txt
import multiCNN.models as models
from multiCNN.utils.general import dic_add
from multiCNN.utils.general import dic_minus
from multiCNN.utils.general import dic_power
from multiCNN.utils.general import dic_div
import multiCNN.utils.transforms as tf
from multiCNN.utils.loss import AutoLearnLossWrapper
from multiCNN.utils.loss import SumLossWrapper
from multiCNN.task_names import TaskNames
from multiCNN.data_management import get_label
from multiCNN.utils.general import show_accession_bbs
from pprint import pprint as pp

import numpy.ma as ma 





class CVTrainer:
    def __init__(self,ts=None,division=5,tag="original",device=None,task_names=None,coral=None,grouping=None,sampling_feature=None,ch_list=None,crop_by_mask_stat=None,crop_by_mask_exec=None,epoch_size=None,
    inception_use_aux_logits=True,shift_loader=None,task_epoch_size=None,shift_stop=None,use_mid_x=False,use_radio_pred_for_diag=False,initialize_zero=False,batch_size=5,
    lambda_dict=None,mid_x_task_names=None,use_aux_task_names=None,mask_normalize=False,use_bce=False,bce_noise_std=None,feature_partition=None,new_diagnosis_cor=None,
    use_multi_loaders=True,imbalance_measure="sampler",not_show_cm=False,lr=None,full_fineturning_epoch=None,lambda_func_dict={},use_imagenet_means_stds=False,input_infos=False,eval_use_infos=False,
    attn_use=False,attn_heads=1,pretrained_path=None,pretrained_imagenet=True,dropout_add=False,use_normalize=True,last_dual=False):
        print("test init start")
        self.ts=ts
        self.division=division
        self.score=[]
        self.state=[]
        self.tag=tag
        # self.radiologic_feature_names=["充実部のwash_out","早期濃染","早期相不均一さ","充実部単純CT濃度"]
        # self.radiologic_feature_names=["早期相不均一さ"]
        # self.radiologic_feature_names=[]
        self.binary=False
        if coral!=None:
            self.coral=coral
        else:
            self.coral=True
        if new_diagnosis_cor==None:
            new_diagnosis_cor=[0,1,2]
        self.new_diagnosis_cor=new_diagnosis_cor
        self.binary_single=False
        self.diagnosis_unique=np.unique(self.new_diagnosis_cor)
        self.num_subtype=len(self.diagnosis_unique)
        # task_names=["diagnosis","充実部単純CT濃度","mask"]
        if task_names==None:
            task_names=["壊死"]
        # task_names=["早期相不均一さ"]
        # task_names=["充実部単純CT濃度"]
        self.tn=TaskNames(task_names,self.binary,num_subtype=self.num_subtype,use_bce=use_bce,use_multi_loaders=use_multi_loaders)
        # self.feature_partition={feature_name:3 for feature_name in self.tn.radiologic_feature_names}
        # if feature_partition==None:
        if feature_partition==None:
            if use_bce:
                feature_partition={feature_name:1 for feature_name in self.tn.radiologic_feature_names}
            else:
                feature_partition={feature_name:3 for feature_name in self.tn.radiologic_feature_names}
        elif type(feature_partition)==int:
            feature_partition_v=feature_partition
            feature_partition={feature_name: feature_partition_v for feature_name in self.tn.radiologic_feature_names}
        self.feature_partition=feature_partition
        self.insert_ch_convert=False
        if ch_list==None:
            ch_list=[ 0,1,3]
        if type(ch_list[0]) is list:
            print("ch_list -> ensemble mode")
            self.ch_ensemble=ch_list
            ch_list=[0,1,2,3]
        else:
            print("ch_list -> normal mode")
            self.ch_ensemble=None
        self.ch_list=ch_list
        self.normalize_range_list=[[-200,400]]*len(self.ch_list)
        # self.crop_by_mask=[0]*3
        self.crop_by_mask_stat=crop_by_mask_stat
        self.crop_by_mask_exec=crop_by_mask_exec
        # home_dir=
        
        # self.grouping="早期相不均一さ"
        # self.sampling_feature="早期相不均一さ"
        if grouping==None:
            grouping="壊死"
        if sampling_feature==None:
            sampling_feature="壊死"
        self.grouping=grouping   # 5分割の指標とする特徴量
        self.sampling_feature=sampling_feature  #オーバーサンプリングするときに参考とする特徴量
        if self.grouping=="first":
            self.grouping=self.tn.loader_out_task_names[0]
        if self.sampling_feature=="first":
            self.sampling_feature=self.tn.loader_out_task_names[0]
        # self.grouping="充実部単純CT濃度"
        # self.sampling_feature="充実部単純CT濃度"
        if shift_loader==None:
            shift_loader=False
        self.shift_loader=shift_loader
        if task_epoch_size==None:
            task_epoch_size=1
        self.task_epoch_size=task_epoch_size
        if shift_stop==None:
            shift_stop=False
        self.shift_stop=shift_stop
        self.loss_wrapper=SumLossWrapper(self.tn.model_task_names,lambda_dict=lambda_dict)
        if device==None:
            self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        self.epoch_size=epoch_size
        if self.epoch_size==None:
            self.epoch_size=350
        if full_fineturning_epoch==None:
            full_fineturning_epoch=100
        self.full_fineturning_epoch=full_fineturning_epoch
        self.input_4ch=(len(self.ch_list))==4
        self.means=None
        self.stds=None
        self.inception_use_aux_logits=inception_use_aux_logits
        task_epoch_size=1
        self.use_mid_x=use_mid_x
        self.mid_x_task_names=mid_x_task_names
        self.use_aux_task_names=use_aux_task_names
        self.use_radio_pred_for_diag=use_radio_pred_for_diag
        self.initialize_zero=initialize_zero
        self.mask_normalize=mask_normalize
        self.use_bce=use_bce
        self.bce_noise_std=bce_noise_std
        self.use_multi_loaders=use_multi_loaders
        self.imbalance_measure=imbalance_measure #(loss or sampler)
        self.not_show_cm=not_show_cm
        self.lambda_dict=lambda_dict
        self.lambda_func_dict=lambda_func_dict
        self.use_imagenet_means_stds=use_imagenet_means_stds
        self.attn_heads=attn_heads
        self.attn_use=attn_use
        self.pretrained_imagenet=pretrained_imagenet
        self.dropout_add=dropout_add
        if lr==None:
            lr=0.0006
        self.lr=lr
        self.batch_size=batch_size
        print("test init finish")

        self.total_dataset=None
        self.pretrained_path=pretrained_path
        self.dt_tag=None
        self.use_normalize=use_normalize
        self.input_infos=input_infos
        self.eval_use_infos=eval_use_infos
        self.last_dual=last_dual
        # self.loss_wrapper=AutoLearnLossWrapper(["diagnosis"]+self.radiologic_feature_names)
    
    '''self.resultの保存形式：trainer.result[クロスバリデーション |0 ][タスク名|"diagnosis"][保存しているデータ名|"predicted_test"][エポック|0]

    '''

    # def set_total_dataset(self):
    #     train_datalist_path=self.dataset_savedir+"/train_multi_features.data"
    #     test_datalist_path=self.dataset_savedir+"/test_multi_features.data"
    #     with open(train_dataset_path, 'rb') as f:
    #         train_accession_list=pickle.load(f)
    #     with open(test_dataset_path, 'rb') as f:
    #         test_accession_list=pickle.load(f)
    #     data_transform_test = tf.data_transform_test_def(img_size,len(self.ch_list),self.normalize_range_list)

    #     accession_list=train_accession_list+test_accession_list
    #     data=Create_Dataset(accession_list,root_path="/data1/RCC/shono_dicom2/npy_2ds",
    #                              data_transform=data_transform_test,spacing_xy=None,spacing_z=1.25,
    #                              ch_list=self.ch_list,z_shift_range=[0,0],
    #                              slice_section=5,
    #                              new_diagnosis_cor=self.new_diagnosis_cor,
    #                              feature_partition=self.feature_partition,
    #                              loader_out_task_names=self.tn.loader_out_task_names,radiologic_feature_names=self.tn.radiologic_feature_names,
    #                              crop_by_mask=self.crop_by_mask_exec,use_bce=self.use_bce,bce_noise_std=None)
    #     self.total_dataset=data


    # def get_img_arr_from_accession_id(self,key_id):

    #     # for accession_id in enumerate(self.accession_path_list:
    #     #     if accession_id==key_id:
    def get_group_id_from_accession_id(self,accession_id):
        task_key=list(res[0].keys())[0]
        for i in range(self.division):
#         print(len(res[i]["diagnosis"]["accession_id_test"][0]))
            if accession_id in self.result[i][task_key]["accession_id_test"][0]:
                return i
        return None
    def get_result_data_from_accession_id(self,accession_id,epoch=None):
        if epoch==None:
            eval_by=["auroc_0_test","auroc_1_test","auroc_2_test"]
            eval_by_feature="diagnosis"
            key="max"
            max_id,score_max,score_max_std,eval_score_max=self.get_score(eval_by,eval_by_feature,key)
            epoch=max_id
        res=self.result
        task_key=list(res[0].keys())[0]
        group_id=-1
        elm_id=-1
        for i in range(self.division):
            for j, accession_id_res in enumerate(res[i][task_key]["accession_id_test"][epoch]):
                if accession_id==accession_id_res:
                    elm_id=j
                    group_id=i
                    break
        res_elm={}
        for task_key in res[0].keys():
            res_elm[task_key]={}
            for attr in res[group_id][task_key].keys():
                res_elm[task_key][attr]=res[group_id][task_key][attr][epoch][elm_id]
        return res_elm
    def show_max_score(self,eval_by="auprc_1_test",eval_by_feature="diagnosis",key="max"):
        print(self.dt_tag)
        if (self.binary and eval_by=="auprc_1_test") or eval_by=="mix":
            eval_by=["sensitivity_0_test","sensitivity_1_test","sensitivity_2_test","specificity_0_test","specificity_1_test","specificity_2_test"]
        if (self.binary and eval_by=="auprc_1_test"):
            eval_by_feature="diagnosis_opt"
        max_id,score_max,score_max_std,eval_score_max=self.get_score(eval_by,eval_by_feature,key)
        print(max_id)
        self.show_score_format(score_max,score_max_std)
    
    def show_score_format(self,score,score_std):
        
        for output_name in score.keys():
            print("\n",output_name)
            avg_max_score={}
            avg_num={}
            for key in score[output_name].keys():
                print(key,score[output_name][key])
                print(key," std ",score_std[output_name][key])
                m=re.fullmatch(r"(\w+)_(\d+)_test",key)
                if m !=None:
                    metric_name=m.group(1)
                    if f"{metric_name}_avg_test" not in avg_max_score.keys():
                        avg_max_score[f"{metric_name}_avg_test"]=score[output_name][key]
                        avg_num[metric_name]=1
                    else:
                        avg_max_score[f"{metric_name}_avg_test"]+=score[output_name][key]
                        avg_num[metric_name]+=1
            for key in avg_max_score.keys():
                avg_max_score[key]/=avg_num[metric_name]
                print(key,avg_max_score[key])
    def plot_score_mean(self,task_name,metric_name):
        score_mean=[]
        for i_elm in range(len(self.score[0][task_name][metric_name])):
            score_elm=0
            for i_chunk in range(len(self.score)):
                score_elm+=self.score[i_chunk][task_name][metric_name][i_elm]
            score_elm/=len(self.score)
            score_mean.append(score_elm)
        # plt.plot(trainer.score[0]["diagnosis"]["sensitivity_2_test"])
        plt.plot(score_mean)


#修正中エリア
    def sum_cv_score(self):
        score_total=None
        for score_division in self.score:
            if score_total==None:
                score_total=cp.copy(score_division)
                if self.not_show_cm:
                    for key in score_total.keys():
                        score_total[key]={k:v for k,v in score_total[key].items() if not "confusion_matrix" in k}
            else:
                if self.not_show_cm:
                    for key in score_division.keys():
                        score_division[key]={k:v for k,v in score_division[key].items() if not "confusion_matrix" in k}
                # print("score_total:",score_total)
                # print("\nscore_division:",score_division)
                score_total=dic_add(score_total,score_division,2)
        score_total=dic_div(score_total,self.division,2)
        score_c=None
        for score_division in self.score:
            if score_c==None:
                score_c_div=cp.copy(score_division)
                score_c=dic_power(dic_minus(score_c_div,score_total,2),2,2)
                
            else:
                score_c_div=cp.copy(score_division)
                score_c_div=dic_add(score_c,dic_power(dic_minus(score_c_div,score_total,2),2,2),2)
        score_c_c=cp.copy(score_c)
        score_std=dic_div(dic_power(dic_div(score_c_c,self.division-1,2),1/2,2),(self.division**(1/2)),2)
        return score_total,score_std

    def get_score_max_id(self,score,eval_by="auprc_1_test",eval_by_feature="diagnosis"):
        if type(eval_by)==list:
            score_eval=[0 for elm in score[eval_by_feature][eval_by[0]]]
            for eval_by_elm in eval_by:
                score_eval=[a1+a2 for a1,a2 in zip(score[eval_by_feature][eval_by_elm],score_eval)]
            eval_score_max=max(score_eval)
            max_id=score_eval.index(eval_score_max)
        else:
            eval_score_max=max(score[eval_by_feature][eval_by])
            max_id=score[eval_by_feature][eval_by].index(eval_score_max)
        return max_id,eval_score_max

    def get_max_epoch_result(self,return_max_id=False):
        eval_by=["sensitivity_0_test","sensitivity_1_test","sensitivity_2_test","specificity_0_test","specificity_1_test","specificity_2_test"]
        eval_by_feature="diagnosis_opt"
        key="max"
        max_id,score_max,score_max_std,eval_score_max=self.get_score(eval_by,eval_by_feature,key)
        print("max_id",max_id)
        data_list_list=[]
        result=self.result
        for pred_type in self.result[0].keys():
            predicted_test=[item for i in range(5) for item in self.result[i][pred_type]["predicted_test"][max_id]]
            predicted_opt_test=[item for i in range(5) for item in self.result[i][pred_type]["predicted_opt_test"][max_id]]
            label_test=[item for i in range(5) for item in self.result[i][pred_type]["label_test"][max_id]]
            probability_test=[item for i in range(5) for item in self.result[i][pred_type]["probability_test"][max_id]]
            accession_id_test=[item for i in range(5) for item in self.result[i][pred_type]["accession_id_test"][max_id]]
            data_list=[{"predicted":predicted,"predicted_opt":predicted_opt,"label":label,"probability":probability,"accession_id":accession_id}for predicted,predicted_opt,label,probability,accession_id in zip(predicted_test,predicted_opt_test,label_test,probability_test,accession_id_test)]
            data_list_list.append(data_list)
        data_t=[]
        data_list_list[0]

        for i in range(len(data_list_list[0])):
            data_t_elm={}
            for i_pred_type,pred_type in enumerate(result[0].keys()):
                data_t_elm[pred_type]=data_list_list[i_pred_type][i]
            data_t.append(data_t_elm)
        if return_max_id:
            return max_id,data_t
        return data_t
    # def 
    # def culc_cond_eval_fp(self,data_t,ref_taskname,taskname,subtype_i,cond_func=None,show_info=False):
    #     if cond_func==None:
    #         cond_func=lambda x: True
    #     x1=0
    #     x2=0

    #     for data_t_elm in data_t:
    #         ref_label=(get_label(data_t_elm[taskname]["accession_id"],ref_taskname)-1)//3
    #         if data_t_elm[taskname]["predicted_opt"]==subtype_i and data_t_elm[taskname]["label"]!=subtype_i and cond_func(ref_label):
    #             x1+=1
    #             if show_info:
    #                 accession_id=data_t_elm[taskname]["accession_id"]
    #                 print("accession:",accession_id)
    #                 print("predicted_opt:",data_t_elm[taskname]["predicted_opt"])
    #                 print("label:",data_t_elm[taskname]["label"])
    #                 print("ref_label:",get_label(data_t_elm[taskname]["accession_id"],ref_taskname))
    #                 show_accession_bbs(accession_id,foot_tag="_registered")

    #         if data_t_elm[taskname]["label"]!=subtype_i and cond_func(ref_label):
    #             x2+=1
    #     if x2==0:
    #         a=0
    #     else:
    #         a=x1/x2
    #     return (a,x1,x2)
    def culc_cond_eval(self,mode,data_t,ref_taskname,taskname,subtype_i,cond_func=None,show_info=False,max_id=None):
        if cond_func==None:
            cond_func=lambda x: True
        x1=0
        x2=0

        for data_t_elm in data_t:
            ref_label=(get_label(data_t_elm[taskname]["accession_id"],ref_taskname)-1)//3
            if mode=="fn":
                mode_cond1=data_t_elm[taskname]["predicted_opt"]!=subtype_i 
                mode_cond2=data_t_elm[taskname]["label"]==subtype_i
            elif mode=="fp":
                mode_cond1=data_t_elm[taskname]["predicted_opt"]==subtype_i 
                mode_cond2=data_t_elm[taskname]["label"]!=subtype_i
            elif mode=="tp":
                mode_cond1=data_t_elm[taskname]["predicted_opt"]==subtype_i 
                mode_cond2=data_t_elm[taskname]["label"]==subtype_i
            elif mode=="tn":
                mode_cond1=data_t_elm[taskname]["predicted_opt"]!=subtype_i 
                mode_cond2=data_t_elm[taskname]["label"]!=subtype_i

            if  mode_cond1 and mode_cond2 and cond_func(ref_label):
                x1+=1
                if show_info:

                    accession_id=data_t_elm[taskname]["accession_id"]

                    print("accession:",accession_id)
                    res_elm=self.get_result_data_from_accession_id(accession_id,max_id)
                    pp(res_elm)
                    print("predicted_opt:",data_t_elm[taskname]["predicted_opt"])
                    print("label:",data_t_elm[taskname]["label"])
                    print("ref_label:",get_label(data_t_elm[taskname]["accession_id"],ref_taskname))
                    if ref_taskname in data_t_elm.keys():
                        print("ref_predicted_opt",data_t_elm[ref_taskname]["predicted_opt"])
                    show_accession_bbs(accession_id,foot_tag="_registered")
                    

            if mode_cond2 and cond_func(ref_label):
                x2+=1
        if x2==0:
            a=0
        else:
            a=x1/x2
        return (a,x1,x2)
    
    def get_score(self,eval_by="auprc_1_test",eval_by_feature="diagnosis",key="max"):
        score_mean,score_std=self.sum_cv_score()
        eval_score_max=None
        if key=="max":
            max_id,eval_score_max=self.get_score_max_id(score_mean,eval_by,eval_by_feature)
            key_id=max_id
        else:
            key_id=key
        score_out={}
        score_out_std={}
        for output_name in score_mean.keys():
            score_out[output_name]={key:value[key_id] for key,value in score_mean[output_name].items()}
            score_out_std[output_name]={key:value[key_id] for key,value in score_std[output_name].items()}
        return key_id,score_out,score_out_std,eval_score_max
    def plot_curves(self,epoch,chunk=None):
        #     epoch=115
        if chunk==None:
            prob_epoch=[]
            label_epoch=[]
            for chunk in range(len(self.result)):
                prob_epoch+=self.result[chunk]["diagnosis"]["probability_test"][epoch]
                label_epoch+=self.result[chunk]["diagnosis"]["label_test"][epoch]
        else:
            prob_epoch=self.result[chunk]["diagnosis"]["probability_test"][epoch]
            label_epoch=self.result[chunk]["diagnosis"]["label_test"][epoch]
        label_list_classes=[]
        prob_list_classes=[]
        for i_class in range(len(prob_epoch[0])):
            prob_list_classes.append([elm[i_class] for elm in prob_epoch] )
            label_list_classes.append([1 if elm==i_class else 0 for elm in label_epoch])
        for i_class in range(len(prob_epoch[0])):
            fpr, tpr, thresholds = roc_curve(label_list_classes[i_class], prob_list_classes[i_class])
            auc=sklearn.metrics.auc(fpr,tpr)
            # auc=roc_auc_score(label_list_classes[i_class], prob_list_classes[i_class])
            print(f"class:{i_class} auc:{auc}")

            plt.plot(fpr, tpr, marker='o')
            plt.xlabel('FPR: False positive rate')
            plt.ylabel('TPR: True positive rate')
            plt.grid()
            plt.show()
            plt.plot(thresholds, tpr, marker='o')
            plt.xlabel('thresholds: thresholds')
            plt.ylabel('TPR: True positive rate')
            plt.grid()
            plt.show()
            plt.plot(thresholds, fpr, marker='o')
            plt.xlabel('thresholds: thresholds')
            plt.ylabel('FPR: False positive rate')
            plt.grid()
            plt.show()
            
    def plot_score(self,score):
        # score_mean,score_std=self.sum_cv_score()
        for output_name in score.keys():
            
            print(output_name)
            if "loss_test" in score[output_name].keys():
                print("loss")
                plt.plot(score[output_name]["loss_test"])
                plt.show()
            if "acc_test" in score[output_name].keys():
                print("acc")

                plt.plot(score[output_name]["acc_test"])
                plt.show()
            if "sensitivity_0_test" in score[output_name].keys():
                num_key=len([key for key in score[output_name].keys() if re.match(r'^sensitivity_\d+_test$',key)])
                # [score[output_name][f'sensitivity_{i_key}_test'] for i_key in range(num_key)]    
                print("sensitivity")
                for i_key in range(num_key):
                    plt.plot(score[output_name][f"sensitivity_{i_key}_test"],label=f"sensitivity_{i_key}_test")
                plt.legend()
                plt.show()

                print("specificity")
                for i_key in range(num_key):
                    plt.plot(score[output_name][f"specificity_{i_key}_test"],label=f"sepecificity_{i_key}_test")
                plt.legend()
                plt.show()
            if "auroc_0_test" in score[output_name].keys():
                num_key=len([key for key in score[output_name].keys() if re.match(r'^auroc_\d+_test$',key)])
                # print("net_trainer_score",net_trainer.score_epoch_list[output_name][f"auroc_{i_key}_test"])
                print("auroc")
                for i_key in range(num_key):
                    plt.plot(score[output_name][f"auroc_{i_key}_test"],label=f"auroc_{i_key}_test")
                plt.legend()
                plt.show()
                print("auprc")
                for i_key in range(num_key):
                    plt.plot(score[output_name][f"auprc_{i_key}_test"],label=f"auprc_{i_key}_test")
                plt.legend()
                plt.show()
            if "mae_0_test" in score[output_name].keys():
                num_key=len([key for key in score[output_name].keys() if re.match(r'^mae_\d+_test$',key)])
                print("mae")
                for i_key in range(num_key):
                    plt.plot(score[output_name][f"mae_{i_key}_test"],label=f"mae_{i_key}_test")
                plt.legend()
                plt.show()
    def plot_cv_score(self):
        score_mean,_=self.sum_cv_score()
        self.plot_score(score_mean)
        
    def dump_result(self,eval_by="auprc_1_test"):
        max_id,score_max,score_max_std,eval_score_max=self.get_score(eval_by)
        score_csvname=f"{self.out_dir}/score_{self.dt_tag}.csv"
        score_max_test={key:value for key,value in score_max.items() if "test" in key}
        score_df=pd.DataFrame.from_dict(score_max_test, orient="index").T
        score_df.to_csv(score_csvname)

    def cross_varidation(self,seed=False):
        import pickle
        import datetime
        if seed:
            seed = 0
            torch.manual_seed(seed)
        self.score=[]
        self.result=[]
        dt_now = datetime.datetime.now()
        dt_tag=dt_now.strftime('%Y_%m_%d__%H_%M_%S')
        self.dt_tag=dt_tag
        out_dir="out_"+self.tag
        self.out_dir=out_dir
        # data_dir="/data1/RCC/shono_dicom2"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        dataset_savedir=out_dir+f"/data_{dt_tag}"
        if not os.path.exists(dataset_savedir):
            os.mkdir(dataset_savedir)
        self.dataset_savedir=dataset_savedir
        

        for i in range(self.division):
            self.execute_chunk(i)
        
        score_filename=f"{out_dir}/score_{dt_tag}.pkl"
        result_filename=f"{out_dir}/result_{dt_tag}.pkl"
        with open(score_filename, "wb") as f:
            pickle.dump(self.score, f)
        with open(result_filename, "wb") as f:
            pickle.dump(self.result, f)

        

    def create_sampler(self,sampling_feature_list):
        sampling_unique,sampling_counts=np.unique(sampling_feature_list,return_counts=True)
        
        class_weights=[sum(sampling_counts)/c for c in sampling_counts]
        # class_weights[2]=0
        print("sp unique",sampling_unique)
        print("class_weight",class_weights)
        print("sampling_feature_list",sampling_feature_list[:20])
        example_weights=[class_weights[e] for e in sampling_feature_list]
        print("class weight",class_weights)
        sampler = WeightedRandomSampler(example_weights,len(sampling_feature_list))
        return sampler

    def culc_class_weights(self,sampling_feature_list):
        sampling_unique,sampling_counts=np.unique(sampling_feature_list,return_counts=True)
        class_weights=[sum(sampling_counts)/c for c in sampling_counts]
        return class_weights

    def execute_chunk(self,test_chunk):
        dataset_path="/data1/RCC/shono_dicom2/useful_accessions_shuffled.txt"
        save_dir="/data1/RCC/shono_dicom2/"
        print("create_data_txt")
        
        Create_dataset_txt(dataset_path,self.dataset_savedir,
        grouping=self.grouping,
        feature_partition=self.feature_partition,
        division=self.division,test_chunk=test_chunk)

        train_datalist_path=self.dataset_savedir+"/train_multi_features.data"
        test_datalist_path=self.dataset_savedir+"/test_multi_features.data"
        model_use="inception"
        lr=self.lr
        momentum=0.9
        batch_size=self.batch_size
        epoch_size=self.epoch_size
        ch_list=self.ch_list
        train_z_shift_range=[-2,2]
        test_z_shift_range=[0,0]
        train_slice_section=5
        test_slice_section=5
        
        
        if model_use=="inception":
            img_size=[299,299]
        elif model_use=="efficient_net":
            img_size=[456,456]
        else:
            img_size=[224,224]
        data_transform = tf.data_transform_train_def(img_size,len(self.ch_list),self.normalize_range_list,use_normalize=False)
        data_transform_test = tf.data_transform_test_def(img_size,len(self.ch_list),self.normalize_range_list,use_normalize=self.use_normalize)
        print("create_dataset!")
        loader_out_task_names_dum=cp.copy(self.tn.loader_out_task_names)
        if not "mask" in self.tn.loader_out_task_names:
            loader_out_task_names_dum.append("mask")
        train_data=Create_Dataset(train_datalist_path,root_path="/data1/RCC/shono_dicom2/npy_2ds",
                                  data_transform=data_transform,spacing_xy=None,spacing_z=1.25,
                                  ch_list=ch_list,z_shift_range=train_z_shift_range,
                                  slice_section=train_slice_section,
                                  new_diagnosis_cor=self.new_diagnosis_cor,
                                  feature_partition=self.feature_partition,
                                  loader_out_task_names=loader_out_task_names_dum,radiologic_feature_names=self.tn.radiologic_feature_names,
                                  crop_by_mask=self.crop_by_mask_stat,use_bce=self.use_bce,bce_noise_std=self.bce_noise_std)
        test_data=Create_Dataset(test_datalist_path,root_path="/data1/RCC/shono_dicom2/npy_2ds",
                                 data_transform=data_transform_test,spacing_xy=None,spacing_z=1.25,
                                 ch_list=ch_list,z_shift_range=test_z_shift_range,
                                 slice_section=test_slice_section,
                                 new_diagnosis_cor=self.new_diagnosis_cor,
                                 feature_partition=self.feature_partition,
                                 loader_out_task_names=self.tn.loader_out_task_names,radiologic_feature_names=self.tn.radiologic_feature_names,
                                 crop_by_mask=self.crop_by_mask_stat,use_bce=self.use_bce,bce_noise_std=None)


        # train_data
        data_transform = tf.data_transform_train_def(img_size,len(self.ch_list),self.normalize_range_list,use_normalize=self.use_normalize)
        print("train_data length:",len(train_data))
        print("train_data shape",train_data[0][1].shape)
        print("test_data length:",len(test_data))

        train_labels_list={task_name:[] for task_name in self.tn.loader_out_task_names}
        ch_num=train_data[0][1].shape[0]
        sum_v=[0]*ch_num
        # total=0
        for labels,image,_ in train_data:
            mask=labels["mask"]>0
            for i_ch in range(ch_num):
                if self.mask_normalize:
                    # print(mask)
                    # print(mask.max())
                    img_ch=torch.masked_select(image[i_ch,:,:], mask)
                else:
                    img_ch=image[i_ch,:,:]
                sum_v[i_ch]+=img_ch.mean()
                # sum_v[i_ch]+=image[i_ch,:,:].mean()
            for task_name in self.tn.loader_out_task_names:
                if type(labels[task_name])==torch.Tensor:
                    train_labels_list[task_name].extend(torch.flatten(labels[task_name]).tolist())
                else:
                    train_labels_list[task_name].append(labels[task_name])
        mean_v=(np.array(sum_v)/len(train_data)).tolist()
        
        total_std=0
        sum_v_std=[0]*ch_num
        for labels,image,_ in train_data:
            mask=labels["mask"]>0
            total_std+=1
            # imgs=item[0]
            for i_ch in range(ch_num):
                if self.mask_normalize:
                    img_ch=torch.masked_select(image[i_ch,:,:], mask)
                else:
                    img_ch=image[i_ch,:,:]
                sum_v_std[i_ch]+=torch.square(img_ch-mean_v[i_ch]).mean()
        std_v=np.sqrt((np.array(sum_v_std)/total_std)).tolist()
        self.mean_v=mean_v
        self.std_v=std_v
        print("means",mean_v)
        print("std_v",std_v)
        if self.use_imagenet_means_stds:
            data_transform = tf.data_transform_train_def(img_size,len(self.ch_list),self.normalize_range_list)
            data_transform_test = tf.data_transform_test_def(img_size,len(self.ch_list),self.normalize_range_list)
        else:
            data_transform = tf.data_transform_train_def(img_size,len(self.ch_list),self.normalize_range_list,means=mean_v,stds=std_v)
            data_transform_test = tf.data_transform_test_def(img_size,len(self.ch_list),self.normalize_range_list,means=mean_v,stds=std_v)
        # data_transform = tf.data_transform_train_def(img_size,len(self.ch_list),self.normalize_range_list)
        # data_transform_test = tf.data_transform_test_def(img_size,len(self.ch_list),self.normalize_range_list)
        train_data.set_data_transform(data_transform)
        test_data.set_data_transform(data_transform_test)
        train_data.crop_by_mask=self.crop_by_mask_exec
        test_data.crop_by_mask=self.crop_by_mask_exec
        train_data.loader_out_task_names=self.tn.loader_out_task_names
        labels_unique={task_name:np.unique(a_task_list) for task_name,a_task_list in train_labels_list.items()}
        train_task_loaders={}
        for feature_name in self.tn.loader_id_task_names:
            if self.use_bce and "diagnosis" not in feature_name:
                train_task_loaders[feature_name]=None
            else:
                train_sampling_feature_list=train_labels_list[feature_name]
                train_sampler=self.create_sampler(train_sampling_feature_list)
                if self.imbalance_measure=="sampler":
                    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle =False,sampler=train_sampler, drop_last=True)
                else:
                    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle =True,sampler=None, drop_last=True)
                train_task_loaders[feature_name]=train_loader
        print("train_task_loaders_id",train_task_loaders.keys())
        print("train_task_loaders_id_names",self.tn.loader_id_task_names)
        train_loader = train_task_loaders[self.sampling_feature]
        print("train_loader_len",len(train_loader))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = False)
        net=models.TNet_2D(model_use=model_use,concat=False,model_task_names=self.tn.model_task_names,
                binary=self.binary,coral=self.coral,
                feature_partition=self.feature_partition,
                num_subtype=self.num_subtype,insert_ch_convert=self.insert_ch_convert,device=self.device,input_4ch=self.input_4ch,
                inception_use_aux_logits=self.inception_use_aux_logits,
                use_mid_x=self.use_mid_x,
                mid_x_task_names=self.mid_x_task_names,
                use_aux_task_names=self.use_aux_task_names,
                use_radio_pred_for_diag=self.use_radio_pred_for_diag,
                initialize_zero=self.initialize_zero,
                use_bce=self.use_bce,eval_use_infos=self.eval_use_infos,
                attn_use=self.attn_use,attn_heads=self.attn_heads,ch_ensemble=self.ch_ensemble,pretrained_path=self.pretrained_path,
                pretrained_imagenet=self.pretrained_imagenet,dropout_add=self.dropout_add,last_dual=self.last_dual)
        net_crop=cp.deepcopy(net)
        net=net.to(self.device)
        net_crop=net_crop.to(self.device)
        if net.device == 'cuda':
            net = torch.nn.DataParallel(net, device_ids=[0, 1]) # make parallel
            net_crop=torch.nn.DataParallel(net_crop, device_ids=[0, 1])
            # torch.backends.cudnn.benchmark=True
        # torch.backends.cudnn.benchmark=False
        loss_wrapper=cp.deepcopy(self.loss_wrapper).to(self.device)
        optimizer = optim.SGD([{"params":net.parameters()},{"params":net_crop.parameters()},{"params":loss_wrapper.parameters()}], lr=lr, momentum=momentum,weight_decay=0.00005)
        
        # criterion = nn.CrossEntropyLoss()
        criterion_dict={}
        for feature_name in self.tn.loader_id_task_names:
            train_sampling_feature_list=train_labels_list[feature_name]
            class_weights=self.culc_class_weights(train_sampling_feature_list)
            if self.imbalance_measure=="loss":
                criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
            else:
                criterion = nn.CrossEntropyLoss()
            criterion=criterion.to(net.device)
            criterion_dict[feature_name]=criterion
        bce_criterion=nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10, eta_min=0.00007,verbose=True)
        ts=trainer.TrainSetting(batch_size=batch_size,epoch_size=epoch_size,optimizer=optimizer,
                criterion=criterion_dict,bce_criterion=bce_criterion,scheduler=scheduler,model_use=model_use,ch_para=False,
                train_loader=train_loader,train_task_loaders=train_task_loaders,test_loader=test_loader,)
        
        net_trainer=trainer.NetTrainer(net,ts,self.tn,
                labels_unique=labels_unique,feature_partition=self.feature_partition,
                binary=self.binary,coral=self.coral,loss_wrapper=loss_wrapper,shift_loader=self.shift_loader,
                shift_stop=self.shift_stop,
                net_crop=net_crop,task_epoch_size=self.task_epoch_size,
                full_fineturning_epoch=self.full_fineturning_epoch,means=mean_v,stds=std_v,
                use_bce=self.use_bce,lambda_dict=self.lambda_dict,lambda_func_dict=self.lambda_func_dict,
                input_infos=self.input_infos)
        net_trainer.train_net()
        self.trainer=net_trainer
        model_savename=self.out_dir+f"/model{test_chunk}_{self.dt_tag}.pth"
        torch.save(net_trainer.net.state_dict(),model_savename)
        self.plot_score(net_trainer.score_epoch_list)
        
        self.score.append(net_trainer.score_epoch_list)
        self.result.append(net_trainer.result_epoch_list)
        self.state.append(net_trainer.state_epoch_list)

