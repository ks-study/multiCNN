import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
import sklearn.metrics
import copy as cp
import torch.nn.functional as F
from torch_optimizer import lamb
from tqdm.notebook import tqdm
import numpy as np
import re
from collections import Counter

from multiCNN.utils.loss import AutoLearnLossWrapper
from multiCNN.utils.loss import SumLossWrapper
from multiCNN.utils.general import get_last_number
from multiCNN.utils.general import show_image_phases

from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import proba_to_label
from sklearn.metrics import mean_absolute_error as mae
from sklearn.utils.class_weight import compute_class_weight

class TrainSetting:
    def __init__(self,batch_size=5,epoch_size=200,optimizer=None,
                 scheduler=None,criterion=None,bce_criterion=None,model_use="inception",ch_para=False,
                train_loader=None,train_task_loaders=None,test_loader=None):
        self.epoch_size=epoch_size

        if criterion==None:
            criterion = nn.CrossEntropyLoss()
        self.optimizer=optimizer
        self.criterion=criterion
        self.scheduler=scheduler
        self.model_use=model_use
        self.ch_para=ch_para
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.train_task_loaders=train_task_loaders
        self.bce_criterion=bce_criterion

class NetTrainer:
    def __init__(
        self,model,train_setting=None,tn=None,
        labels_unique={"diagnosis":[0,1,2],"早期濃染":[0,1,2]},
        feature_partition=None,binary=True,coral=False,loss_wrapper=None
        ,shift_loader=False,net_crop=None,task_epoch_size=1,shift_stop=False,
        full_fineturning_epoch=None,means=None,stds=None,use_bce=False,lambda_dict=None,lambda_func_dict={},input_infos=False):

        if train_setting==None:
            train_setting=TrainSetting()
        self.ts=train_setting
        self.score_epoch_list={}
        self.state_epoch_list=[]
        self.labels_unique=labels_unique
        self.net=model
        self.net_crop=net_crop
        self.tn=tn
        self.is_multitask=len(self.tn.radiologic_feature_names)>0
        self.binary=binary
        self.coral=coral
        if "diagnosis" in self.labels_unique.keys():
            self.num_subtype=len(self.labels_unique["diagnosis"])
        self.shift_loader=shift_loader
        if loss_wrapper==None:
            loss_wrapper=SumLossWrapper(self.tn.model_task_names)
        self.loss_wrapper=loss_wrapper
        self.task_epoch_size=task_epoch_size
        self.shift_stop=shift_stop
        if feature_partition==None:
            feature_partition={feature_name: 3 for feature_name in self.radiologic_feature_names}
        self.feature_partition=feature_partition
        self.full_fineturning_epoch=full_fineturning_epoch
        print("feature_partition:",feature_partition)
        if isinstance(self.net,nn.DataParallel):
            self.device=self.net.module.device
        else:
            self.device=self.net.device
        # self.means=means
        # self.stds=stds
        self.use_bce=use_bce
        self.lambda_dict=lambda_dict
        self.lambda_func_dict=lambda_func_dict
        self.input_infos=input_infos
        

    def set_setting(self,train_setting):
        self.ts=train_setting

    def predict_diagnosis_optimal(self,threshold,prob):
        if type(threshold)==list:
            predicted=len(prob)-1
            for i_type,prob_type in enumerate(prob[:-1]):
                if prob_type>=threshold[i_type]:
                    predicted=i_type
                    break
        else:
            prob_cc=prob[0]
            if prob_cc>=threshold:
                predicted=0
            else:
                prob_others=prob[1:]
                predicted=prob_others.index(max(prob_others))+1
        
        return predicted
    def predict_coral_optimal(self,thresholds,prob):
        predicted=0
        for threshold_class,prob_class in zip(thresholds,prob):
            if prob_class>=threshold_class:
                predicted+=1
        return predicted
    def predict_diagnosis_optimal_bin(self,threshold_bins,prob_bins):
        predicted=self.num_subtype-1
        for bin_i,(prob,threshold) in enumerate(zip(prob_bins,threshold_bins)):
            if prob[1]>=threshold[1]:
                predicted=bin_i
                break
        return predicted

    def generate_predicted_list_bin(self,predicted_list_bins):
        pred_list=[]
        for i in range(len(predicted_list_bins[0])):
            pred_bins=[pred_bin_elm[i] for pred_bin_elm in predicted_list_bins]
            pred=len(predicted_list_bins)-1
            for i_bin,pred_bin_elm in enumerate(pred_bins):
                if pred_bin_elm[1]>0.5:
                    pred=i_bin
                    break
            pred_list.append(pred)
        return pred_list

    def confusion_matrix(self,label_list,predicted_list,labels_unique):
        result_counts_template={"TP":0,"TN":0,"FP":0,"FN":0}
        result_counts={}
        for label in labels_unique:
            result_counts[label]=result_counts_template.copy()
        for elm_i,(label_i,predicted_i) in enumerate(zip(label_list,predicted_list)):
            for label_ref in labels_unique:
                positive=0
                true_=0
                label_positive=0
                if predicted_i==label_ref:
                    positive=1
                if label_i==label_ref:
                    label_positive=1
                if label_positive==positive:
                    true_=1
                if positive:
                    if true_:
                        result_counts[ label_ref]["TP"]+=1
                    else:
                        result_counts[ label_ref]["FP"]+=1
                else:
                    if true_:
                        result_counts[ label_ref]["TN"]+=1
                    else:
                        result_counts[ label_ref]["FN"]+=1
        return result_counts

    def process_roc(self,label_list_ref, proba_list_ref,label_num_list):
        fpr,tpr,thresholds = sklearn.metrics.roc_curve(label_list_ref, proba_list_ref)
        J = tpr -fpr
        ix = np.argmax(J)
        fpr_optimal=fpr[ix]
        tpr_optimal=tpr[ix]
        threshold_optimal=thresholds[ix]
        # threshold_optimal
        total_num=len(label_list_ref)
        acc_optimal=(tpr_optimal*label_num_list[1]+(1-fpr_optimal)*label_num_list[0])/total_num
        
        precision_c, recall_c, thresholds_pr =sklearn.metrics.precision_recall_curve(label_list_ref, proba_list_ref)
        auroc=sklearn.metrics.auc(fpr,tpr)
        auprc=sklearn.metrics.auc(recall_c,precision_c)
        return auroc,auprc,acc_optimal,threshold_optimal,tpr_optimal,fpr_optimal

    def process_roc_label(self,labels_unique, probability_list,label_list,predicted_list):
        auroc_label=[]
        auprc_label=[]
        fpr_optimal_label=[]
        tpr_optimal_label=[]
        acc_optimal_label=[]
        thresholds_label=[]
        for label_ref in labels_unique:
            proba_list_ref=[proba[label_ref] for proba in probability_list]
            label_list_ref=[ 1 if label==label_ref else 0 for label in label_list]
            predict_list=[1 if pred==label_ref else 0 for pred in predicted_list]
            label_num_list=[label_list_ref.count(0),label_list_ref.count(1)]
            auroc,auprc,acc_optimal,threshold_optimal,tpr_optimal,fpr_optimal=self.process_roc(label_list_ref, proba_list_ref,label_num_list)
            fpr_optimal_label.append(fpr_optimal)
            tpr_optimal_label.append(tpr_optimal)
            acc_optimal_label.append(acc_optimal)
            auroc_label.append(auroc)
            auprc_label.append(auprc)
            thresholds_label.append(threshold_optimal)
        return fpr_optimal_label,tpr_optimal_label,acc_optimal_label,auroc_label,auprc_label,thresholds_label

    def culc_sp_coral(self,label_list,predicted_list,task_name):
        # future_partition_num=9//self.feature_partition[task_name]
        if self.feature_partition[task_name]>=3:
            label_list_3=[elm//(self.feature_partition[task_name]//3) for elm in label_list]
            predicted_list_3=[elm//(self.feature_partition[task_name]//3) for elm in predicted_list]
        labels_unique=[0,1,2]
        acc_feature,data_epoch_feature=self.culc_sp(label_list_3,predicted_list_3,None,task_name=task_name,labels_unique=labels_unique)
        
        return acc_feature,data_epoch_feature

    def culc_sp(self,label_list,predicted_list,probability_list,task_name=None,labels_unique=None):
        if labels_unique==None:
            labels_unique=self.labels_unique[task_name]
        result_counts=self.confusion_matrix(label_list,predicted_list,labels_unique)
        
        precision_label=[]
        sensitivity_label=[]
        specificity_label=[]
        IoU_label=[]
        f1_label=[]
        for label_ref in labels_unique:
            if (result_counts[label_ref]["TP"]==0):
                precision=0.0
                sensitivity=0.0
                IoU=0.0
            else:
                precision=result_counts[label_ref]["TP"]/(result_counts[label_ref]["TP"]+result_counts[label_ref]["FP"])
                sensitivity=result_counts[label_ref]["TP"]/(result_counts[label_ref]["TP"]+result_counts[label_ref]["FN"])
                IoU=result_counts[label_ref]["TP"]/(result_counts[label_ref]["TP"]+result_counts[label_ref]["FN"]+result_counts[label_ref]["FP"])
            if (result_counts[label_ref]["TN"]==0):
                specificity=0.0
            else:
                specificity=result_counts[label_ref]["TN"]/(result_counts[label_ref]["TN"]+result_counts[label_ref]["FP"])
            # if 
            precision_label.append(precision)
            sensitivity_label.append(sensitivity)
            specificity_label.append(specificity)
            IoU_label.append(IoU)
            # f1_label.append(f1)
        correct=0
        total=0
        for label_i,predicted_i in zip(label_list,predicted_list):
            if label_i==predicted_i:
                correct+=1
            total+=1
        acc=0.0
        if correct>0:
            acc=correct/total
        if probability_list!=None:
            fpr_optimal_label,tpr_optimal_label,acc_optimal_label,auroc_label,auprc_label,thresholds_label=self.process_roc_label(labels_unique, probability_list,label_list,predicted_list)
            data_epoch_label={
                "precision":precision_label,
                "sensitivity":sensitivity_label,
                "specificity":specificity_label,
                "tpr_optimal":tpr_optimal_label,
                "fpr_optimal":fpr_optimal_label,
                "acc_optimal":acc_optimal_label,
                "auroc":auroc_label,
                "auprc":auprc_label,
                "threshold":thresholds_label
            }
            # if return_threshold:

            #     return acc,data_epoch_label,thresholds_label
        else:
            data_epoch_label={
                "precision":precision_label,
                "sensitivity":sensitivity_label,
                "specificity":specificity_label,
            }
        if task_name=="mask":
            data_epoch_label["IoU"]=IoU_label
        return acc,data_epoch_label
    def culc_metric_mae(self,label_list,predicted_list,probability_list,task_name=None,labels_unique=None):
        if labels_unique==None:
            labels_unique=self.labels_unique[task_name]
        mae_label=[]
        mae_label=[mae(label_list,probability_list)]
        data_epoch_label={
            "mae":mae_label,
        }
        acc=0.0
        return acc,data_epoch_label

    def culc_sp_mask(self,label_list,predicted_list,probability_list):
        # print(label_list[0])
        # label_list_mask=[np.array(elm).ravel() for elm in label_list]
        label_list_mask=list(np.array(label_list).reshape(-1,1))
        predicted_list=list(np.array(predicted_list).reshape(-1,1))
        # print("np_shape",np.array(probability_list).shape)
        probability_list=list(np.array(probability_list).reshape(-1,1))
        # predicted_list=[np.array(elm).ravel() for elm in predicted_list]
        # probability_list=[np.array(elm).ravel() for elm in probability_list]
        acc,data_epoch_label=self.culc_sp(label_list_mask,predicted_list,probability_list,task_name="mask",labels_unique=[0])
        # print(data_epoch_label)
        return acc,data_epoch_label
    def loss_inception(self,criterion,labels,out,aux_out=None):
        if aux_out!=None:
            loss1 = criterion(out,labels)
            loss2 = criterion(aux_out,labels)
            loss=loss1 + 0.4*loss2
        else:
            loss=criterion(out,labels)
        return loss

    def culc_loss(self,labels_data,out_dict,aux_out_dict=None,train=False,labels_coral=None,mask_label=None):
        ts=self.ts
        loss_tasks_dict={}
        for task_name in self.tn.model_task_names:
            if task_name=="diagnosis":
                if aux_out_dict!=None:
                    loss_task=self.loss_inception(ts.criterion[task_name],labels_data[task_name],out_dict[task_name],aux_out_dict[task_name])
                else:
                    loss_task=ts.criterion[task_name](out_dict[task_name],labels_data[task_name])
            elif "diagnosis_bin" in task_name:
                if aux_out_dict!=None:
                    loss_task=self.loss_inception(ts.criterion[task_name],labels_data[task_name],out_dict[task_name],aux_out_dict[task_name])
                else:
                    loss_task=ts.criterion[task_name](out_dict[task_name],labels_data[task_name])
            elif task_name=="mask":
                mask_criterion= nn.BCELoss()
                # mask_criterion= nn.CrossEntropyLoss()
                mask_label=labels_data["mask"]
                mask_label_deform=mask_label.view(mask_label.shape[0],-1)
                out_mask_flatten=out_dict[task_name].view(mask_label_deform.shape[0],-1)
                if aux_out_dict!=None:
                    aux_out_mask_flatten=aux_out_dict[task_name].view(mask_label_deform.shape[0],-1)
                # print("mask_label_shape",mask_label.shape)
                # print("mask_shape",mask_label_deform.shape)
                # print("out_mask_shape",out_mask_flatten.shape)
                if aux_out_dict!=None:
                    
                    # mask_label_deform=mask_label
                    
                    loss_task=self.loss_inception(mask_criterion,mask_label_deform,F.sigmoid(out_mask_flatten),F.sigmoid(aux_out_mask_flatten))
                else:
                    loss_task=mask_criterion(F.sigmoid(out_mask_flatten),mask_label_deform)
            else:
                if self.coral:
                    if aux_out_dict!=None:
                        loss_task=self.loss_inception(coral_loss,labels_coral[task_name],out_dict[task_name],aux_out_dict[task_name])
                    else:
                        loss_task=coral_loss(out_dict[task_name],labels_coral[task_name])
                else:
                    if self.use_bce:
                        criterion=ts.bce_criterion
                        out_dict_task=torch.squeeze(out_dict[task_name])
                        if aux_out_dict!=None:
                            aux_out_dict_task=torch.squeeze(aux_out_dict[task_name])
                        labels_data_task=labels_data[task_name].float()
                    else:
                        criterion=ts.criterion[task_name]
                        out_dict_task=out_dict[task_name]
                        if aux_out_dict!=None:
                            aux_out_dict_task=aux_out_dict[task_name]
                        labels_data_task=labels_data[task_name]
                    # print("out_dict",out_dict_task)
                    # print("labels_data_task",labels_data_task)
                    if aux_out_dict!=None:
                        # print("out_shape",out_dict[task_name].shape)
                        # print("labels_shape",labels_data[task_name].shape)
                        loss_task=self.loss_inception(criterion,labels_data_task,out_dict_task,aux_out_dict_task)
                    else:
                        loss_task=criterion(out_dict_task,labels_data_task)
            loss_tasks_dict[task_name]=loss_task
        loss=self.loss_wrapper(loss_tasks_dict)
        return loss,loss_tasks_dict

    def exec_batch(self,data,batch_id,epoch,train=False,loader_id_task_name=None):


        # batch_train_task_names=self.tn.model_task_names
        if self.shift_loader:
            if train and loader_id_task_name!=None:
                loss_lambda_dict={task_name:(1 if task_name==loader_id_task_name else 0) for task_name in self.tn.model_task_names}
                self.loss_wrapper=SumLossWrapper(self.tn.model_task_names,loss_lambda_dict)
            else:
                    self.loss_wrapper=SumLossWrapper(self.tn.model_task_names)
        labels_data,inputs,accession_ids=data
        

        
        # show_image_phases(labels_data["mask"][0])
        # show_image_phases(inputs[0],means=self.means,stds=self.stds)
        inputs=inputs.to(self.device,dtype=torch.float)
        # mask=labels_data["mask"].to(self.device,dtype=torch.float)
        
        # inputs=list(torch.chunk(inputs,inputs.shape[1],dim=1))
        # if ts.ch_para:
        #     for i in range(len(inputs)):
        #         inputs[i]=torch.cat((inputs[i],inputs[i],inputs[i]), dim=1)
        # else:
        #     inputs[0]=torch.cat([inputs[i_ch] for i_ch in range(len(inputs))],dim=1)
        #     for i in range(len(inputs)):
        #         inputs[i]=inputs[0]
        labels_data={task_name:value.to(self.device, dtype=torch.long) for task_name,value in labels_data.items()}
        if self.input_infos:
            input_infos=labels_data
        else:
            input_infos=None
        if "mask" in labels_data.keys():
            labels_data["mask"]=labels_data["mask"].float()
            labels_data["mask"]=F.max_pool2d(labels_data["mask"],kernel_size=16)
        
        
        # labels_data["mask"]=mask_label_deform.view(mask_label_deform.shape[0],-1)
        # if "mask" in self.tn.model_task_names:
        #     ロス関数の設定が必要

        #     pass
        labels_coral=None
        if self.coral:
            labels_coral={}

            for task_name in self.tn.radiologic_feature_names:
                # feature_partition_num=9//self.feature_partition[task_name]
                labels_coral[task_name]=levels_from_labelbatch(labels_data[task_name], num_classes=self.feature_partition[task_name]).to(self.device, dtype=torch.long)
        if train:
            self.ts.optimizer.zero_grad()
        
        #モデルの推測、Lossの計算
        if self.ts.model_use=="inception" and train and self.net.inception_use_aux_logits:
            out_dict,aux_out_dict=self.net(inputs,input_infos)
            
            loss,loss_batch_dict=self.culc_loss(labels_data,out_dict,aux_out_dict,train=True,labels_coral=labels_coral)
        
        else:
            # loss_batch_dict={}
            # print("",inputs[0].shape)
            out_dict=self.net(inputs,input_infos)
            loss,loss_batch_dict=self.culc_loss(labels_data,out_dict,aux_out_dict=None,train=False,labels_coral=labels_coral)
            #diagnosisのlossを計算
            # for task_name in self.tn.model_task_names:
            #     if task_name=="diagnosis":
            #         loss_task = ts.criterion(out_dict[task_name],labels_data[task_name])
            #     elif "diagnosis_bin" in task_name:
            #         loss_task = ts.criterion(out_dict[task_name],labels_data[task_name])
            #     else:
            #         if self.coral:
            #             # labels_coral=levels_from_labelbatch(labels_data[task_name], num_classes=self.feature_partition[task_name]).to(self.device, dtype=torch.long)
            #             loss_task =coral_loss(out_dict[task_name],labels_coral[task_name])
            #             # if task_name in labels_coral_epoch_dict.keys():
            #             #     labels_coral_epoch_dict[task_name]+=labels_coral.cpu().detach().numpy().copy().tolist()
            #             # else:
            #             #     labels_coral_epoch_dict[task_name]=labels_coral.cpu().detach().numpy().copy().tolist()
            #         else:
            #             loss_task = ts.criterion(out_dict[task_name],labels_data[task_name])
            #     loss_tasks_dict[task_name]=loss_task
            # loss=self.loss_wrapper(loss_tasks_dict)
        # print("loss_auto state:",self.loss_wrapper.state_dict())
        #Lossの逆伝播
        if train:
            loss.backward()
            self.ts.optimizer.step()
        #
        if epoch%10==0:
            if batch_id%10==0:
                if "mask" in self.tn.model_task_names:
                    print(f"epoch: {epoch} batch: {batch_id}")
                    # im_arr=inputs[0][0].cpu().numpy()
                    im_arr=inputs[0]
                    print("is",im_arr.shape)
                    show_image_phases(im_arr)
                    show_image_phases(labels_data["mask"][0])
                    show_image_phases(out_dict["mask"][0])
                    show_image_phases(out_dict["mask"][0]>0.5)
        #予測確率、予測値、ラベルをresult_data_epochに保存
        result_data_batch={task_name:{} for task_name in self.tn.result_task_names}
        correct_batch_dict={}
        for task_name in self.tn.model_task_names:
            labels_task=None
            if "mask" == task_name:
                probability=out_dict[task_name].data
                predicted=out_dict[task_name].data>=0.5
                # probability=probability.reshape(probability.shape[0],-1)
                # predicted=predicted.reshape(predicted.shape[0],-1)
                # label_list_mask=[elm.flatten() for elm in probability]
                # print("mask_shape",labels_data["mask"].shape)
                # print("pred_shape",predicted.shape)
                # corects=(predicted == labels_data[task_name])
                # print("corect_sum",corects.sum().item())
                # print((predicted == labels_data[task_name]).shape)
                # labels_task=
            elif "diagnosis_bin" in task_name:
                predicted=torch.max(out_dict[task_name].data, 1)[1]
                probability=F.softmax(out_dict[task_name].data)
            elif task_name=="diagnosis":
                predicted=torch.max(out_dict[task_name].data,1)[1]
                probability=F.softmax(out_dict[task_name])
            else:
                if self.coral:
                    probability=torch.sigmoid(out_dict[task_name])
                    predicted=proba_to_label(probability).float()
                else:
                    if self.use_bce:
                        probability=torch.sigmoid(out_dict[task_name])
                        predicted=torch.squeeze(probability.data>=0.5)
                        
                    else:
                        predicted=torch.max(out_dict[task_name], 1)[1]
                        probability=F.softmax(out_dict[task_name])
            correct_batch_dict[task_name] = (predicted == labels_data[task_name]).sum().item()
            if task_name in  self.tn.result_task_names:
                result_data_batch[task_name]["probability"]=probability.cpu().detach().numpy().copy().tolist()
                result_data_batch[task_name]["predicted"]=predicted.cpu().detach().numpy().copy().tolist()
                result_data_batch[task_name]["label"]=labels_data[task_name].cpu().detach().numpy().copy().tolist()
                result_data_batch[task_name]["accession_id"]=accession_ids
        # if self.binary:
            
        #     result_data_batch["diagnosis"]["predicted"]=predicted.cpu().detach().numpy().copy().tolist()
        #     result_data_batch["diagnosis"]["label"]=labels_data["diagnosis"].cpu().detach().numpy().copy().tolist()
        #     result_data_batch["diagnosis"]["accession_id"]=accession_ids

            

        if self.binary and "diagnosis" in self.tn.result_task_names:
            # print("good")
            # self.diagnosis_bin_task_names
            pred_bins=[0]*len(self.tn.diagnosis_bin_task_names)
            for task_name in self.tn.diagnosis_bin_task_names:
                i_bin=get_last_number(task_name)
                pred_bins[i_bin]=np.array(result_data_batch[task_name]["predicted"])
            # print("testes",pred_bins[0])
            # for 
            predicted_diagnosis=np.zeros([len(pred_bins[0])])
            for bin_i,pred_bin in reversed(list(enumerate(pred_bins))):
                predicted_diagnosis=np.where(pred_bin==1,bin_i,predicted_diagnosis)
            result_data_batch["diagnosis"]["predicted"]=predicted_diagnosis.tolist()
            result_data_batch["diagnosis"]["label"]=labels_data["diagnosis"].cpu().detach().numpy().copy().tolist()
            result_data_batch["diagnosis"]["accession_id"]=accession_ids
        if self.coral:
            return (loss,loss_batch_dict,correct_batch_dict,result_data_batch,labels_data,labels_coral)
        else:
            return (loss,loss_batch_dict,correct_batch_dict,result_data_batch,labels_data)

    def culc_thresholds_coral(self,labels_coral,probability_list):
        # labels_coral=labels_coral_epoch_dict[origin_task_name]
        labels_coral_trans=[[]]*len(labels_coral[0])
        prob_coral_trans=[[]]*len(labels_coral[0])
        pred_coral_trans=[[]]*len(labels_coral[0])
        for labels_coral_sample,prob_coral_sample in zip(labels_coral,probability_list):
            for i_class in range(len(labels_coral_sample)):
                labels_coral_trans[i_class].append(labels_coral_sample[i_class])
                # prob_coral_trans[i_class].append([1-prob_coral_sample[i_class],prob_coral_sample[i_class]])
                prob_coral_trans[i_class].append([1-prob_coral_sample[i_class],prob_coral_sample[i_class]])
                pred_coral_trans[i_class].append(int(prob_coral_sample[i_class]>=0.5))
        thresholds_coral=[]
        for i_class in range(len(labels_coral_trans)):
            # print("prob",prob_coral_trans[i_class][:5])
            # print("pred",pred_coral_trans[i_class][:5])
            # print("label",labels_coral_trans[i_class][:5])
            acc_task,data_epoch_label_task=self.culc_sp(labels_coral_trans[i_class],pred_coral_trans[i_class],prob_coral_trans[i_class],labels_unique=[0,1])
            thresholds_coral.append(data_epoch_label_task["threshold"][1])

        return thresholds_coral

    def generate_predict_list_with_threshold_selection(self,origin_task_name,probability_data,thresholds_data):
        predicted_list_opt=[]
        if self.binary and origin_task_name=="diagnosis":
            # prob_bins_epoch=[result_data_epoch[bin_task_name]["probability"] for bin_task_name in self.tn.diagnosis_bin_task_names ]
            # threshold_bins_epoch=[data_epoch_label_task[bin_task_name]["threshold"] for bin_task_name in self.tn.diagnosis_bin_task_names ]
            # probability_data
            for prob_bins_elm in range(len(probability_data)):
                #indice: 一つのモデルが出力する確率の数（二値分類なら2つ。）
                # prob_bins_elm=[prob_bins[indice] for prob_bins in probability_data]
                # label_bins=[label_bins[indice] for label_bins in label_bins_epoch]
                predicted=self.predict_diagnosis_optimal_bin(thresholds_data,prob_bins_elm)
                predicted_list_opt.append(predicted)
        else:
            if self.coral and origin_task_name in self.tn.radiologic_feature_names:
                for prob_coral_sample in probability_data:
                    predicted=self.predict_coral_optimal(thresholds_data,prob_coral_sample )
                    predicted_list_opt.append(predicted)
            else:
                thresholds=thresholds_data
                # print("thresholds",thresholds)
                for prob in probability_data:
                    # print("prob",prob)
                    predicted=self.predict_diagnosis_optimal(thresholds,prob)
                    predicted_list_opt.append(predicted)
        return predicted_list_opt
        

    def exec_epoch(self,epoch,train=False,loader_id_task_name=None):
        # print("epoch_start")
        if len(self.lambda_func_dict)!=0:
            lambda_dict_epoch=cp.copy(self.lambda_dict)
            for task_name,lambda_func in self.lambda_func_dict.items():
                lambda_dict_epoch[task_name]=lambda_func(epoch,self.lambda_dict[task_name])
            self.loss_wrapper=SumLossWrapper(self.tn.model_task_names,lambda_dict_epoch)
            # self.loss_wrapper:
        #------1.各変数の初期化
        ts=self.ts
        if train:
            if self.shift_loader:
                if loader_id_task_name!=None:
                    dataloader=ts.train_task_loaders[loader_id_task_name]
                else:
                    dataloader=ts.train_task_loaders[self.tn.loader_id_task_names[0]]
            else:
                dataloader=ts.train_loader
            
            self.net.train()
            self.net_crop.train()
        else:
            dataloader=ts.test_loader
            self.net.eval()
            self.net_crop.eval()
        state="train" if train else "test"
        result_data_epoch_feature={"accession_id":[],"label":[],"predicted":[],"probability":[]}
        result_data_epoch={
            task_name:cp.deepcopy(result_data_epoch_feature) for task_name in self.tn.result_task_names
        }
        loss_epoch_test=0
        predicted_list=[]
        probability_list=[]
        labels_list=[]
        total=0
        correct=0
        correct_epoch_dict={task_name:0 for task_name in self.tn.result_task_names}
        loss_epoch=0
        loss_epoch_diagnosis=0
        loss_epoch_features=0
        loss_epoch_dict={}
        loss_epoch_dict.update({task_name:0 for task_name in self.tn.model_task_names})
        labels_coral_epoch_dict={}
        #-------2. バッチを回して学習
        for batch_i,data in enumerate(tqdm(dataloader,leave=False)):
            if self.coral:
                loss,loss_batch_dict,correct_batch_dict,result_data_batch,labels_data,labels_coral=self.exec_batch(data,batch_i,epoch,train=train,loader_id_task_name=None)
            else:
                loss,loss_batch_dict,correct_batch_dict,result_data_batch,labels_data=self.exec_batch(data,batch_i,epoch,train=train,loader_id_task_name=None)
            # print("keys",correct_batch_dict.keys())
            for task_name in correct_batch_dict.keys():
                correct_epoch_dict[task_name] += correct_batch_dict[task_name]
            if self.coral:
                for task_name in labels_coral.keys():
                    if task_name in labels_coral_epoch_dict.keys():
                        labels_coral_epoch_dict[task_name]+=labels_coral[task_name].cpu().detach().numpy().copy().tolist()
                    else:
                        labels_coral_epoch_dict[task_name]=labels_coral[task_name].cpu().detach().numpy().copy().tolist()
            loss_epoch+=loss
            loss_epoch_dict={key:val+loss_batch_dict[key] for key,val in loss_epoch_dict.items()}
            total+=labels_data[self.tn.model_task_names[0]].size(0)
            # print("batch_keys",result_data_batch.keys())
            for task_name in result_data_batch.keys():
                for key in result_data_batch[task_name].keys():
                    result_data_epoch[task_name][key]+=result_data_batch[task_name][key]
        ts.scheduler.step()


        #------3. 評価値の計算（閾値選択なし）
        acc_dict={task_name:correct_elm/total for task_name,correct_elm in correct_epoch_dict.items()}
        # acc=correct/total
        data_epoch_label_tasks={}
        acc_tasks={}
        cm_tasks={}
        for task_name in self.tn.model_task_names:
            if task_name not in self.tn.eval_task_names:
                continue
            if "diagnosis_bin" in task_name:
                acc_task,data_epoch_label_task=self.culc_sp(result_data_epoch[task_name]["label"],result_data_epoch[task_name]["predicted"],result_data_epoch[task_name]["probability"],task_name=task_name)
                
                # data_epoch_diagnosis_bins.append(data_epoch_diagnosis_bin)
            elif "diagnosis"==task_name:
                if "probability" in result_data_epoch[task_name].keys():
                    probability_list=result_data_epoch[task_name]["probability"]
                else:
                    probability_list=None
                acc_task,data_epoch_label_task=self.culc_sp(result_data_epoch[task_name]["label"],result_data_epoch[task_name]["predicted"],probability_list,task_name=task_name)
            elif task_name=="mask":
                acc_task,data_epoch_label_task=self.culc_sp_mask(result_data_epoch[task_name]["label"],result_data_epoch[task_name]["predicted"],result_data_epoch[task_name]["probability"])
            else:
                if self.coral:
                    acc_task,data_epoch_label_task=self.culc_sp_coral(result_data_epoch[task_name]["label"],result_data_epoch[task_name]["predicted"],task_name=task_name)
                elif self.use_bce:
                    acc_task,data_epoch_label_task=self.culc_metric_mae(result_data_epoch[task_name]["label"],result_data_epoch[task_name]["predicted"],result_data_epoch[task_name]["probability"],task_name=task_name)
                    print(task_name,data_epoch_label_task)
                else:
                    acc_task,data_epoch_label_task=self.culc_sp(result_data_epoch[task_name]["label"],result_data_epoch[task_name]["predicted"],result_data_epoch[task_name]["probability"],task_name=task_name)

            acc_tasks[task_name]=acc_task
            if task_name!="mask" :
                if not self.use_bce or "diagnosis" in task_name:
                    cm_tasks[task_name]=sklearn.metrics.confusion_matrix(result_data_epoch[task_name]["label"],result_data_epoch[task_name]["predicted"])
            data_epoch_label_tasks[task_name]=data_epoch_label_task
        
        # print("dataepoch_labels",data_epoch_label_tasks.keys())
        if self.binary:
            # prob_bin_data=[result_data_epoch[task_name]["probability"] for task_name in self.tn.diagnosis_bin_task_names]
            
            # predicted_list=self.generate_predicted_list_bin(prob_bin_data)
            label_list=result_data_epoch["diagnosis"]["label"]
            pred_list=result_data_epoch["diagnosis"]["predicted"]
            acc_task,data_epoch_label_task=self.culc_sp(label_list,pred_list,None,task_name="diagnosis")
            acc_tasks["diagnosis"]=acc_task
            data_epoch_label_tasks["diagnosis"]=data_epoch_label_task
        #------4.閾値選択をして評価
        #------4.1. coral用のthresholdsを計算
        opt_task_names=[task_name for task_name in self.tn.eval_task_names if "_opt" in task_name]
        thresholds_coral_dict={}
        
        if self.coral:
            for task_name in self.tn.radiologic_feature_names:
                labels_coral=labels_coral_epoch_dict[task_name]
                # labels_coral_task_name=
                probability_list=result_data_epoch[task_name]["probability"]
                thresholds_coral=self.culc_thresholds_coral(labels_coral,probability_list)
                thresholds_coral_dict[task_name]=thresholds_coral
        # if self.binary and "diagnosis" in self.tn.eval_task_names:
        #     prob_bins_epoch=[result_data_epoch[bin_task_name]["probability"] for bin_task_name in self.tn.diagnosis_bin_task_names ]
        #     threshold_bins_epoch=[data_epoch_label_task[bin_task_name]["threshold"] for bin_task_name in self.tn.diagnosis_bin_task_names ]
        
        #------4.2. 閾値選択による評価値の算出
        
        for opt_task_name in opt_task_names:
            # if "diagnosis_opt" in self.tn.eval_task_names:
            origin_task_name=opt_task_name[:-4]
            # if self.coral and origin_task_name in self.radiologic_feature_names:
                
            if self.binary and origin_task_name=="diagnosis":
                # probability_data=prob_bins_epoch
                probability_data=[result_data_epoch[bin_task_name]["probability"] for bin_task_name in self.tn.diagnosis_bin_task_names ]
                probability_data=list(map(list,zip(*probability_data)))
                thresholds_data=[data_epoch_label_tasks[bin_task_name]["threshold"] for bin_task_name in self.tn.diagnosis_bin_task_names ]
            else:
                probability_data=result_data_epoch[origin_task_name]["probability"]
                if origin_task_name=="mask":
                    probability_data=list(np.array(probability_data).reshape(-1,1))
                if self.coral and origin_task_name in self.tn.radiologic_feature_names:
                    thresholds_data=thresholds_coral_dict[origin_task_name]
                else:
                    thresholds_data=data_epoch_label_tasks[origin_task_name]["threshold"]
            # print("origin_task_name",origin_task_name)
            
            predicted_list_opt=self.generate_predict_list_with_threshold_selection(origin_task_name,probability_data,thresholds_data)
            # probability_data_epoch=
            # task_name="diagnosis_opt"
            if origin_task_name=="mask":
                # culc_sp_mask(self,label_list,predicted_list,probability_list)
                acc_task,data_epoch_label_task=self.culc_sp_mask(result_data_epoch[origin_task_name]["label"],predicted_list_opt,None)
            elif self.coral and not "diagnosis" in origin_task_name:
                acc_task,data_epoch_label_task=self.culc_sp_coral(result_data_epoch[origin_task_name]["label"],predicted_list_opt,task_name=origin_task_name)

            else:
                acc_task,data_epoch_label_task=self.culc_sp(result_data_epoch[origin_task_name]["label"],predicted_list_opt,None,task_name=origin_task_name)
            data_epoch_label_tasks[opt_task_name]=data_epoch_label_task
            result_data_epoch[origin_task_name]["predicted_opt"]=predicted_list_opt
            if task_name!="mask":
                cm_tasks[opt_task_name]=sklearn.metrics.confusion_matrix(result_data_epoch[origin_task_name]["label"],predicted_list_opt)
            acc_tasks[opt_task_name]=acc_task

        #------5.data_epochへ格納
        data_epoch={}
        for task_name in self.tn.eval_task_names:

            data_epoch[task_name]={"acc":acc_tasks[task_name]}
            if task_name in loss_epoch_dict.keys():
                data_epoch[task_name]["loss"]=loss_epoch_dict[task_name]
            # data_epoch_feature=data_epoch_features[feature_name]
            for key,data_epoch_label_elm in data_epoch_label_tasks[task_name].items():
                # print(key,data_epoch_label_elm)
                if type(data_epoch_label_elm)==list:
                    for label_i,data_epoch_task_i in enumerate(data_epoch_label_elm):
                        data_epoch[task_name][f"{key}_{label_i}"]=data_epoch_task_i
                else:
                    data_epoch[task_name][key]=data_epoch_label_elm
            if task_name in cm_tasks.keys():
                data_epoch[task_name]["confusion_matrix"]=cm_tasks[task_name]
        # print(data_epoch)
        # state_epochにタスク共通の値を保存
        if len(self.loss_wrapper.state_dict())>0:
            loss_wrapper_params=self.loss_wrapper.state_dict()["log_vars"].cpu()
        else:
            loss_wrapper_params=0
        state_epoch={"loss_params":loss_wrapper_params}

        # for eval_name in data_epoch["mask"].keys():
        #     print(eval_name)
        #     print(data_epoch["mask"][eval_name])
        return data_epoch,result_data_epoch,state_epoch

    def show_data_epoch(self,data_epoch,epoch):
        # for key
        def get_eval_score_format(tag):
            num_key=len([key for key in data_epoch.keys() if re.match(r'^{}_\d+$'.format(tag),key)])
            # print()
            # return "{}:{} ".format(tag,[f'{tag}_{i_key}' for i_key in range(num_key)])
            return "{}:{} ".format(tag,np.round([data_epoch[f'{tag}_{i_key}'] for i_key in range(num_key)],4))
        
        # print(data_epoch.keys())
        result="epoch:{}".format(epoch)
        if "acc" in data_epoch.keys():
            # print("acc",data_epoch['acc'])
            result+="acc:{:.4} ".format(data_epoch['acc'])
        if "loss" in data_epoch.keys():
            result+="loss:{:.4} ".format(data_epoch['loss'])
        if "sensitivity_0" in data_epoch.keys():
            result+=get_eval_score_format("sensitivity")+get_eval_score_format("specificity")
        if "auroc_0" in data_epoch.keys():
            result+=get_eval_score_format("tpr_optimal")+get_eval_score_format("fpr_optimal") \
                +get_eval_score_format("auroc")+get_eval_score_format("auroc")
        if "mae_0"  in data_epoch.keys():
            result+=get_eval_score_format("mae")
        print(result)

    def train_net(self):
        self.score_epoch_list={}
        self.result_epoch_list={}
        ts=self.ts
        i_loader_task=0
        for epoch in range(ts.epoch_size):
            if self.full_fineturning_epoch!=None:
                if epoch==self.full_fineturning_epoch:
                    for param in self.net.parameters():
                        param.requires_grad = True
            if self.shift_loader:
                # task_epoch_size=80
                loader_id_task_name=self.tn.loader_id_task_names[i_loader_task]
                i_loader_task=(epoch//self.task_epoch_size)%len(self.tn.loader_id_task_names)
                if epoch>=self.task_epoch_size*len(self.tn.loader_id_task_names) and self.shift_stop:
                    data_epoch_train,result_data_epoch_train,state_epoch=self.exec_epoch(epoch,train=True)
                else:
                    data_epoch_train,result_data_epoch_train,state_epoch=self.exec_epoch(epoch,train=True,loader_id_task_name =loader_id_task_name)
            else:
                data_epoch_train,result_data_epoch_train,state_epoch=self.exec_epoch(epoch,train=True)
            print("train")
            for data_epoch_key in data_epoch_train.keys():
                print(f"{data_epoch_key}: ")
                self.show_data_epoch(data_epoch_train[data_epoch_key],epoch)
            with torch.no_grad():
                data_epoch_test,result_data_epoch_test,state_epoch=self.exec_epoch(epoch,train=False)
            print("test")
            for data_epoch_test_key in data_epoch_test.keys():
                print(f"{data_epoch_test_key}: ")
                self.show_data_epoch(data_epoch_test[data_epoch_test_key],epoch)
            if epoch==0:
                self.state_epoch_list=[]
                for output_name in data_epoch_train.keys():
                    self.score_epoch_list[output_name]={}
                    for key in data_epoch_train[output_name].keys():
                        self.score_epoch_list[output_name][f"{key}_train"]=[]
                        self.score_epoch_list[output_name][f"{key}_test"]=[]
                for output_name in result_data_epoch_test.keys():
                    self.result_epoch_list[output_name]={}
                    for key in result_data_epoch_test[output_name].keys():
                        self.result_epoch_list[output_name][f"{key}_test"]=[]
            self.state_epoch_list.append(state_epoch)
            for output_name in data_epoch_train.keys():
                for key in data_epoch_train[output_name].keys():
                    self.score_epoch_list[output_name][f"{key}_train"].append(data_epoch_train[output_name][key])
                    self.score_epoch_list[output_name][f"{key}_test"].append(data_epoch_test[output_name][key])
                if output_name in result_data_epoch_test.keys():
                    for key in result_data_epoch_test[output_name].keys():
                        self.result_epoch_list[output_name][f"{key}_test"].append(result_data_epoch_test[output_name][key])
            # cm=self.score_epoch_list["diagnosis"][f"confusion_matrix_test"]
            # print(cm)
            

            