import torch

def masked_mae_cal(inputs,target,mask):
    return torch.sum(torch.abs(inputs-target)*mask)/(torch.sum(mask)+1e-9)

def masked_mse_cal(inputs,target,mask):
    return torch.sum(torch.square(inputs-target)*mask)/(torch.sum(mask)+1e-9)

def masked_rmse_cal(inputs,target,mask):
    return torch.sqrt(masked_mse_cal(inputs,target,mask))

def masked_mre_cal(inputs,target,mask):
    return torch.sum(torch.abs(inputs-target)*mask)/(torch.sum(torch.abs(target*mask))+1e-9)
