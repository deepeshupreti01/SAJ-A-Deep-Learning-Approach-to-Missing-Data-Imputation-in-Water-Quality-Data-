import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils import masked_mae_cal


class FeatureRegression(nn.Module):
  def __init__(self,input_size:int):
    super().__init__()
    self.W = nn.Parameter(torch.Tensor(input_size,input_size))
    self.b = nn.Parameter(torch.Tensor(input_size))
    m = torch.ones(input_size,input_size)-torch.eye(input_size)
    self.register_buffer('m',m)
    self.reset_param()

  def reset_param(self):
    std = 1./math.sqrt(self.W.size(0))
    self.W.data.uniform_(-std,std)
    if self.b is not None:
      self.b.data.uniform_(-std,std)


  def forward(self,x):
    return F.linear(x,self.W*Variable(self.m),self.b)

class TemporalDecay(nn.Module):
  def __init__(self,input_size:int,output_size:int,diag:bool=False):
    super().__init__()
    self.diag = diag
    self.W = nn.Parameter(torch.Tensor(output_size,input_size))
    self.b = nn.Parameter(torch.Tensor(output_size))
    if self.diag:
      assert (input_size==output_size),'If diagnoal then input size must equal with output size'
      m = torch.eye(input_size)
      self.register_buffer('m',m)
    self.reset_param()

  def reset_param(self):
    std = 1./math.sqrt(self.W.size(0))
    self.W.data.uniform_(-std,std)
    if self.b is not None:
      self.b.data.uniform_(-std,std)


  def forward(self,deltas:torch.Tensor):
    if self.diag:
      gamma = F.relu(F.linear(deltas,self.W*Variable(self.m),self.b))
    else:
      gamma = F.relu(F.linear(deltas,self.W,self.b))
    return torch.exp(-gamma)

class RITS(nn.Module):
  def __init__(self,hidden_size:int,seq_len:int,n_features:int,dropout:float,eps:float=1e-9,MIT:bool=False,device:str="cpu"):
    super().__init__()
    self.eps = eps
    self.seq_len = seq_len
    self.hidden_size = hidden_size
    self.rnn = nn.LSTMCell(n_features*2,hidden_size)
    self.tempdecay_h = TemporalDecay(n_features,hidden_size)
    self.tempdecay_x = TemporalDecay(n_features,n_features,True)
    self.hist_reg = nn.Linear(hidden_size,n_features)
    self.feat_reg = FeatureRegression(n_features)
    self.combining_weights = nn.Linear(n_features*2,n_features)
    self.dropout = nn.Dropout(dropout)
    self.MIT = MIT
    self.device = device


  def forward(self,data,direction):
    values = data[direction]['X']
    mask = data[direction]['missing_mask']
    deltas = data[direction]['deltas']
    h,c = torch.zeros((values.size()[0],self.hidden_size),device=self.device),torch.zeros((values.size()[0],self.hidden_size),device=self.device)
    x_loss = 0.
    mae_loss = 0.
    imputations = []
    for t in range(self.seq_len):
      x = values[:,t,:]
      m = mask[:,t,:]
      d = deltas[:,t,:]
      gamma_h = self.tempdecay_h(d)
      gamma_x = self.tempdecay_x(d)
      h = h*gamma_h
      x_h = self.hist_reg(h)
      x_loss += masked_mae_cal(x_h,x,m)
      x_c = m*x + (1-m)*x_h
      z_h = self.feat_reg(x_c)
      x_loss += masked_mae_cal(z_h,x,m)
      alpha = self.combining_weights(torch.cat([gamma_x,m],dim=1))
      c_h = alpha*z_h + (1-alpha)*x_h
      mae_loss += masked_mae_cal(c_h,x,m)
      x_loss += mae_loss
      c_c = m*x + (1-m)*c_h
      inputs = torch.cat([c_c,m],dim=1)
      h,c = self.rnn(inputs,(h,c))
      imputations.append(c_h.unsqueeze(dim=1))

    imputations = torch.cat(imputations,dim=1)
    imputed_data = mask*values+(1-mask)*imputations
    x_loss /= self.seq_len*3
    mae_loss /= self.seq_len
    ret_dict = {
      "consistency_loss":torch.tensor(0.,device=self.device),
      'reconstruction_loss':x_loss,
      "reconstruction_MAE":mae_loss,
      "imputed_data":imputed_data
    }
    if "X_holdout" in data:
      ret_dict["X_holdout"] = data["X_holdout"]
      ret_dict["indicating_mask"] = data["indicating_mask"]
    return ret_dict

class BRITS(nn.Module):
  def __init__(self,rits_forward:RITS,rits_backward:RITS):
    super().__init__()
    self.rits_forward = rits_forward
    self.rits_backward = rits_backward

  def consistency_loss(self,forward_pred,backward_pred):
    loss = torch.abs(forward_pred-backward_pred).mean()*1e-1
    return loss

  def _forward(self,data):
    return self.rits_forward(data,"forward")

  def _backward(self,data):
    return self.rits_backward(data,"backward")

  def _reverse(self,ret):
    def reverse_tensor(tensor_):
      if tensor_.dim()<=1:
        return tensor_
      indices = range(tensor_.size()[1])[::-1]
      indices = torch.tensor(indices,dtype=torch.long,device=tensor_.device,requires_grad=False)
      return tensor_.index_select(1,indices)

    for value in ret:
      ret[value] = reverse_tensor(ret[value])

    return ret

  def merge_ret(self,ret_f,ret_b):
    consistency_loss = self.consistency_loss(ret_f['imputed_data'],ret_b['imputed_data'])
    imputed_data = (ret_f['imputed_data']+ret_b['imputed_data'])/2
    reconstruction_loss = (ret_f['reconstruction_loss']+ret_b['reconstruction_loss'])/2
    reconstruction_MAE = (ret_f['reconstruction_MAE']+ret_b['reconstruction_MAE'])/2
    ret_f['imputed_data'] = imputed_data
    ret_f['consistency_loss'] = consistency_loss
    ret_f['reconstruction_loss'] =  reconstruction_loss
    ret_f['reconstruction_MAE'] = reconstruction_MAE
    return ret_f

def build_BRITS(seq_len,feature_num,hidden_size,dropout,MIT,device)->BRITS:
  rits_f = RITS(hidden_size,seq_len,feature_num,dropout,MIT=MIT,device=device)
  rits_b = RITS(hidden_size,seq_len,feature_num,dropout,MIT=MIT,device=device)
  brits = BRITS(rits_f,rits_b)
  return brits
