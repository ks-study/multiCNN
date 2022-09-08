
# from pygame import BLENDMODE_NONE
import torch.nn as nn
import torch
from torchvision import models
from efficientnet_pytorch import EfficientNet
import copy as cp
import random
from coral_pytorch.layers import CoralLayer
import multiCNN.my_inception as my_inception
import torch.nn.functional as F
from multiCNN.utils.general import load_backbone_state_dict
from coral_pytorch.dataset import levels_from_labelbatch


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
def inception_3ch_to_4ch(model_ft,initialize_zero=False):
    new_in_channels = 4
    layer=model_ft.Conv2d_1a_3x3.conv
    new_layer=nn.Conv2d(4, layer.out_channels, kernel_size=layer.kernel_size, 
    stride=layer.stride, bias=layer.bias)
    copy_weights = 0 # Here will initialize the weights from new channel with the red channel weights
    # Copying the weights from the old to the new layer
    with torch.no_grad():
        new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()
        weight_padding=layer.weight[:, copy_weights:copy_weights+1, : :]
        if initialize_zero:
            weight_padding=torch.zeros_like(weight_padding)
            # weight_padding=layer.weight[:, copy_weights:copy_weights+1, : :]
        #Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
        for i in range(new_in_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight[:, channel:channel+1, :, :] = weight_padding.clone()
            # new_layer.weight[:, channel:channel+1, :, :] = layer.weight[:, copy_weights:copy_weights+1, : :].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)
        # model_ft.Conv2d_1a_3x3.conv
        model_ft.Conv2d_1a_3x3.conv = new_layer
        for param in model_ft.parameters():
            param.requires_grad = False
    first_layer = list(model_ft.children())[0]
    # last_layer = list(model_ft.children())[-1]
    # print(f'except last layer: {first_layer}')
    for param in first_layer.parameters():
        param.requires_grad = True
    return model_ft

class TNet_2D(nn.Module):
    def __init__(self,model_use="inception",concat=False,
    model_task_names=["早期濃染"],binary=False,coral=False,feature_partition=None,
    num_subtype=3,insert_ch_convert=False,device=None,inception_use_aux_logits=True,input_4ch=False,
    use_mid_x=False,use_radio_pred_for_diag=False,initialize_zero=False,mid_x_task_names=None,use_aux_task_names=None,use_bce=False,attn_use=False,attn_heads=1,
    ch_ensemble=None,eval_use_infos=False,pretrained_path=None,pretrained_imagenet=True,dropout_add=False,last_dual=False):
        super(TNet_2D, self).__init__()
        self.model_task_names=model_task_names
        self.radiologic_feature_names=[name for name in self.model_task_names if "diagnosis" not in name ]
        self.model_use=model_use
        self.ch_convert=nn.Conv2d(4,3,3,1,1)
        self.use_radio_pred_for_diag=use_radio_pred_for_diag
        self.use_mid_x=use_mid_x
        self.use_bce=use_bce
        self.binary=binary
        self.inception_use_aux_logits=inception_use_aux_logits
        self.middle_out=False
        self.ch_ensemble=ch_ensemble #ex: [[0,1,2],[1,2,3]]
        if mid_x_task_names==None:
            mid_x_task_names=self.radiologic_feature_names
        self.mid_x_task_names=mid_x_task_names
        if use_aux_task_names==None:
            use_aux_task_names=self.model_task_names
        self.use_aux_task_names=use_aux_task_names
        if insert_ch_convert:
            self.insert_ch_convert=True
        else:
            self.insert_ch_convert=False
        if model_use=="efficient_net":
            model_ft = EfficientNet.from_pretrained('efficientnet-b5') 
            model_ft._fc = Identity()
        elif model_use=="res_net":
            model_ft = models.resnet152(pretrained=True)
            model_ft.fc = Identity()
        elif model_use=="inception":
            
            # model_ft=models.inception_v3(pretrained=True,aux_logits=inception_use_aux_logits)
            
            # else:
            print("pretrained_imagenet:",pretrained_imagenet)
            model_ft=my_inception.inception_v3(pretrained=pretrained_imagenet,aux_logits=inception_use_aux_logits,transform_input=False,attn_use=attn_use,attn_heads=attn_heads,dropout_add=dropout_add)
            if input_4ch and ch_ensemble==None:
                model_ft=inception_3ch_to_4ch(model_ft,initialize_zero=initialize_zero)
            if pretrained_path!=None:
                param=torch.load(pretrained_path,map_location="cuda:0")
                # model_ft.load_state_dict(param["state_dict"])
                load_backbone_state_dict(model_ft, param["state_dict"])
                # model_ft.load_state_dict()
            model_ft.fc = Identity()
            if inception_use_aux_logits:
                model_ft.AuxLogits.fc=Identity()
        elif model_use=="efficient_net_3ch":
            model_ft = EfficientNet.from_pretrained('efficientnet-b5') 
            model_ft._fc = nn.Linear(2048, 5)
        
        
        self.conv_phase=[]
        if device==None:
            self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        self.conv_phase0=model_ft
        self.conv_phase1=cp.deepcopy(model_ft)
        self.conv_phase2=cp.deepcopy(model_ft)
        self.conv_phase3=cp.deepcopy(model_ft)
        if ch_ensemble!=None:
            self.conv_phase_list=nn.ModuleList([self.conv_phase0,self.conv_phase1,self.conv_phase2,self.conv_phase3])
        self.mp=nn.MaxPool2d((2,2))
        self.fc1=nn.Linear(15,2)
        self.drop=nn.Dropout(p=0.08)
        self.concat=concat
        self.eval_use_infos=eval_use_infos
        if feature_partition==None:
            if self.use_bce:
                feature_partition={feature_name:1 for feature_name in self.radiologic_feature_names}
            else:
                feature_partition={feature_name:3 for feature_name in self.radiologic_feature_names}
        self.feature_partition=feature_partition
        # self.fc_diagnosis=nn.Linear(2048,3)
        # self.aux_fc_diagnosis=nn.Linear(768,3)
        # # self.fc_diagnosis_bin0=nn.Linear(2048,2)
        # # self.fc_diagnosis_bin1=nn.Linear(2048,2)
        # self.fc_diagnosis_bins=nn.ModuleList([nn.Linear(2048,2) for i_subtype in range(num_subtype)])
        # self.aux_fc_diagnosis_bins=nn.ModuleList([nn.Linear(768,2) for i_subtype in range(num_subtype)])
        # self.aux_fc_diagnosis_bin0=nn.Linear(768,2)
        # self.aux_fc_diagnosis_bin1=nn.Linear(768,2)
        fc_list=[]
        aux_fc_list=[]
        mid_out_num=10368
        out_features_num=sum(feature_partition.values())
        print("out feature num",out_features_num)
        
        if coral:
            out_features_num-=len(list(feature_partition.values()))
        print("out feature num2",out_features_num)
        # out_features_num=sum(feature_partition.values())
        # out_features_num=3*len(self.radiologic_feature_names)
        for task_name in self.model_task_names:
            fc_input_num=2048
            fc_aux_input_num=768
            if self.ch_ensemble!=None:
                fc_input_num*=len(self.ch_ensemble)
                fc_aux_input_num*=len(self.ch_ensemble)
            if "diagnosis" in task_name and self.use_radio_pred_for_diag:
                fc_input_num+=out_features_num
                print("fc_input",fc_input_num)
            if task_name in self.mid_x_task_names and self.use_mid_x:
                fc_input_num=mid_out_num
            print(task_name,fc_input_num)
            if "_bin" in task_name:
                layer=nn.Linear(fc_input_num,2)
                if inception_use_aux_logits:
                    layer_aux=nn.Linear(fc_aux_input_num,2)
            elif "mask"== task_name:
                up= nn.Upsample(size=[18,18], mode='bilinear', align_corners=True)
                # up= nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                deconv = DoubleConv(fc_input_num,3, 1024)
                layer=nn.Sequential(
                    up,deconv,
                )
            elif coral and task_name in self.radiologic_feature_names:
                print(f"coral layer {task_name}")
                layer=CoralLayer(fc_input_num,feature_partition[task_name])
                if inception_use_aux_logits:
                    layer_aux=CoralLayer(fc_aux_input_num,feature_partition[task_name])
            elif self.use_bce and task_name in self.radiologic_feature_names:
                layer=nn.Linear(fc_input_num, 1)
                if inception_use_aux_logits:
                    layer_aux=nn.Linear(fc_aux_input_num,1)
            else:
                if task_name=="diagnosis":
                    output_num=num_subtype
                else:
                    output_num=feature_partition[task_name]
                if last_dual:
                    layer=nn.Sequential(nn.Linear(fc_input_num,1024),self.drop,nn.Linear(1024,output_num))
                else:
                    layer=nn.Linear(fc_input_num,output_num)
                
                if inception_use_aux_logits:
                    layer_aux=nn.Linear(fc_aux_input_num,output_num)
            fc_list.append([task_name,layer])
            if inception_use_aux_logits:
                aux_fc_list.append([task_name,layer_aux])
        self.fc_dict=nn.ModuleDict(fc_list)
        self.features_mode=False
        self.coral=coral
        if inception_use_aux_logits:
            self.aux_fc_dict=nn.ModuleDict(aux_fc_list)
        
    
    def set_middle_out(self,middle_out):
        # self.features_mode=features_mode
        self.middle_out=middle_out
        self.conv_phase0.middle_out=middle_out
        self.conv_phase1.middle_out=middle_out
        self.conv_phase2.middle_out=middle_out
        self.conv_phase3.middle_out=middle_out
        
    def set_features_mode(self,features_mode):
        self.features_mode=features_mode
        self.conv_phase0.features_mode=features_mode
        self.conv_phase1.features_mode=features_mode
        self.conv_phase2.features_mode=features_mode
        self.conv_phase3.features_mode=features_mode
    def forward(self,image0,input_infos=None):
        return self._forward(image0,self.features_mode,input_infos=input_infos)
    def _forward(self,image0,features_mode,input_infos=None):
        # print("forwarding data")
        # print("img0",image0.shape)
        # print("img0",image0)
        if self.insert_ch_convert:
            image0=self.ch_convert(image0)
        diag_task_names=[task_name for task_name in self.model_task_names if "diagnosis" in task_name]
        feature_task_names=[task_name for task_name in self.model_task_names if not "diagnosis" in task_name]
        model_task_names_sorted=feature_task_names+diag_task_names
        # print("image_shape",image0.shape)
        if self.training and self.model_use=="inception" and self.inception_use_aux_logits:
            # print()
            # a=self.conv_phase0(image0)
            if self.ch_ensemble!=None:
                input_list=[]
                for i,layer_ids in enumerate(self.ch_ensemble):
                    input_list_elm=torch.zeros_like(image0[:,:3])
                    # image0[]
                    for j,layer_id in enumerate(layer_ids):
                        input_list_elm[:,j]=image0[:,layer_id]
                    input_list.append(input_list_elm)
                x=None
                for i,image_i in enumerate(input_list):
                    if x==None:
                        x,e_out,mid_x,aux_x,aux_e=self.conv_phase_list[i](image_i)
                    else:
                        x_i,e_out_i,mid_x_i,aux_x_i,aux_e_i=self.conv_phase_list[i](image_i)
                        # print("shapes",x.shape,e_out.shape,mid_x.shape,aux_x.shape,aux_e.shape)
                        # print("shapes",x_i.shape,e_out_i.shape,mid_x_i.shape,aux_x_i.shape,aux_e_i.shape)
                        x=torch.cat((x,x_i),axis=1)
                        e_out=torch.cat((e_out,e_out_i),axis=1)
                        mid_x=torch.cat((mid_x,mid_x_i),axis=1)
                        aux_x=torch.cat((aux_x,aux_x_i),axis=1)
                        aux_e=torch.cat((aux_e,aux_e_i),axis=1)
            else:
                # print("image_shape",image0.shape)
                if self.middle_out:
                    return self.conv_phase0(image0)
                x,e_out,mid_x,aux_x,aux_e=self.conv_phase0(image0)
            if features_mode:
                return x,e_out,mid_x,aux_x,aux_e
            out_dict={}
            aux_out_dict={}
            
            for task_name in model_task_names_sorted:
                if "diagnosis" in task_name and self.use_radio_pred_for_diag:
                    cat_list=[]
                    
                    for out_dict_key,out_dict_elm in out_dict.items() :
                        if input_infos!=None and out_dict_key in input_infos.keys() and not self.use_bce:
                            if self.coral:
                                out_e=levels_from_labelbatch(input_infos[out_dict_key], num_classes=self.feature_partition[out_dict_key]).to(self.device)
                            else:
                                out_e=F.one_hot(input_infos[out_dict_key], num_classes=self.feature_partition[out_dict_key])
                            cat_list+=[out_e]
                        else:
                            cat_list+=[out_dict_elm]
                    x_with_radio=torch.cat(tuple([x]+cat_list),-1)
                    # print("x with radio",x_with_radio.shape)
                    out_dict[task_name]=self.fc_dict[task_name](x_with_radio)
                    if task_name in self.use_aux_task_names:
                        aux_out_dict[task_name]=self.aux_fc_dict[task_name](aux_x)
                    else:
                        aux_out_dict[task_name]=self.fc_dict[task_name](x)
                    # aux_out_dict[task_name]=self.aux_fc_dict[task_name](aux_x)
                elif task_name=="mask":
                    
                    out_dict[task_name]=self.fc_dict[task_name](e_out)
                    aux_out_dict[task_name]=out_dict[task_name]
                else:
                    if self.use_mid_x and task_name in self.mid_x_task_names:
                        # print("Mid out shape",mid_x.shape)
                        out_dict[task_name]=self.fc_dict[task_name](mid_x)
                    else:
                        # print("out shape",x.shape)
                        out_dict[task_name]=self.fc_dict[task_name](x)
                    if task_name in self.use_aux_task_names:
                        aux_out_dict[task_name]=self.aux_fc_dict[task_name](aux_x)
                    else:
                        aux_out_dict[task_name]=out_dict[task_name]
            return out_dict,aux_out_dict
        else:

            if self.ch_ensemble!=None:
                input_list=[]
                for i,layer_ids in enumerate(self.ch_ensemble):
                    input_list_elm=torch.zeros_like(image0[:,:3])
                    # image0[]
                    for j,layer_id in enumerate(layer_ids):
                        input_list_elm[:,j]=image0[:,layer_id]
                    input_list.append(input_list_elm)
                x=None
                for i,image_i in enumerate(input_list):
                    if x==None:
                        x,e_out,mid_x=self.conv_phase_list[i](image_i)
                    else:
                        x_i,e_out_i,mid_x_i=self.conv_phase_list[i](image_i)
                        x=torch.cat((x,x_i),axis=1)
                        e_out=torch.cat((e_out,e_out_i),axis=1)
                        mid_x=torch.cat((mid_x,mid_x_i),axis=1)
            else:
                if self.middle_out:
                    return self.conv_phase0(image0)
                x,e_out,mid_x=self.conv_phase0(image0)

            if features_mode:
                return x,e_out,mid_x
            out_dict={}
            # aux_out_dict={}
            for task_name in model_task_names_sorted:
                if "diagnosis" in task_name and self.use_radio_pred_for_diag:
                    cat_list=[]
                    for out_dict_key,out_dict_elm in out_dict.items() :
                        if input_infos!=None and out_dict_key in input_infos.keys() and not self.use_bce and self.eval_use_infos:
                            if self.coral:
                                out_e=levels_from_labelbatch(input_infos[out_dict_key], num_classes=self.feature_partition[out_dict_key]).to(self.device)
                            else:
                                out_e=F.one_hot(input_infos[out_dict_key], num_classes=self.feature_partition[out_dict_key])
                            cat_list+=[out_e]
                        else:
                            cat_list+=[out_dict_elm]
                    x_with_radio=torch.cat(tuple([x]+cat_list),-1)
                    out_dict[task_name]=self.fc_dict[task_name](x_with_radio)
                elif task_name=="mask":
                    out_dict[task_name]=self.fc_dict[task_name](e_out)
                else:
                    if self.use_mid_x and task_name in self.mid_x_task_names:
                        out_dict[task_name]=self.fc_dict[task_name](mid_x)
                    else:
                        out_dict[task_name]=self.fc_dict[task_name](x)
            return out_dict

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#単なる確率分布に従った当てずっぽう
class Unigram:
    def __init__(self,counts):
        self.counts=counts
        self.dice=range(len(counts))
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.device
    def predict(self,k=1):
        return torch.tensor(random.choices(self.dice, k=k,weights = self.counts)).to(self.device)

class EnsembleModel(nn.Module):
    def __init__(self,model_list,convert_func_list=None):
        self.model_list=nn.ModuleList(model_list)
        self.convert_func_list=convert_func_list
        for convert_func in self.convert_func_list:
            if type(convert_func)==list:
                convert_func=lambda phase_list : tuple([phase_list[phase_id] for phase_id in convert_func ])

    
    def forward(self,*args):
        model_output_list=[]
        for model,convert_func in zip(self.model_list,self.convert_func_list):
            model_input=convert_func(args)
            model_output=model(model_input)
            model_output_list.append(model_output)
        model_output_total=None

        for model_output in model_output_list:
            if type(model_output)==tuple:
                if model_output_total==None:
                    model_output_total=model_output
                else:
                    for model_output_e,model_output_total_e in model_output,model_output_total:
                        for model_output_task_name,model_output_task in model_output_e.items():
                            model_output_total_e[model_output_task_name]+=model_output_task
            else:
                if model_output_total==None:
                    model_output_total=model_output
                else:
                    for model_output_task_name,model_output_task in model_output.items():
                        model_output_total[model_output_task_name]+=model_output_task
        if type(model_output)==tuple:
            for model_output_total_e in model_output_total:
                for model_output_task_name,model_output_total_e_task in model_output_total_e.items():
                    model_output_total_e_task/=len(self.model_list)
        else:
            for model_output_task_name,model_output_total_task in model_output_total.items():
                model_output_total_task/=len(self.model_list)
        return model_output_total


            
