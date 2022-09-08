import torch.nn as nn
import torch

class AutoLearnLossWrapper(nn.Module):
    def __init__(self, output_names):
        super(AutoLearnLossWrapper, self).__init__()
        self.output_names = output_names
        self.task_num=len(output_names )
        self.log_vars = nn.Parameter(torch.zeros((self.task_num)))

    def forward(self, loss_dict):
        loss=0
        for loss_elm,log_var_elm in zip(loss_dict.values(),self.log_vars):
            precision = 1/2*torch.exp(-2*log_var_elm)
            loss+=loss_elm*precision+log_var_elm
        return loss

class SumLossWrapper(nn.Module):
    def __init__(self, output_names,lambda_dict=None):
        super(SumLossWrapper, self).__init__()
        if lambda_dict==None:
            lambda_dict={}
        for output_name in output_names:
            
            if output_name not in lambda_dict.keys():
                lambda_dict[output_name]=1
        self.lambda_dict=lambda_dict

    def forward(self, loss_dict):
        loss=0
        for loss_name,loss_elm in loss_dict.items():
            loss+=self.lambda_dict[loss_name]*loss_dict[loss_name]
        return loss
