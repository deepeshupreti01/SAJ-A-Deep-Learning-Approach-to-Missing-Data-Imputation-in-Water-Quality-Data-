import math
import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
  def __init__(self,d_features:int,d_model:int):
    super().__init__()
    self.linear = nn.Linear(d_features,2*d_model)

  def forward(self,x):
    return self.linear(x)

class PositionalEncoding(nn.Module):
  def __init__(self,d_model:int,seq_len:int,dropout:float):
    super().__init__()
    self.d_model =2*d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)

    pe = torch.zeros(seq_len,self.d_model)
    position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0,self.d_model,2).float()*(-math.log(10000.)/self.d_model))
    pe[:,0::2] = torch.sin(position*div_term)
    pe[:,1::2] = torch.cos(position*div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe',pe)

  def forward(self,x):
    x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
    return self.dropout(x)

class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, d_model: int, d_v:int, h: int, dropout: float):
    super().__init__()
    self.d_v = 2*d_v
    self.d_model = 2*d_model
    self.h = h
    assert self.d_model % h == 0, "d_model is not divisible by h"
    self.d_k = self.d_model // h
    self.w_q = nn.Linear(self.d_model, self.d_k*self.h, bias=False)
    self.w_k = nn.Linear(self.d_model, self.d_k*self.h, bias=False)
    self.w_v = nn.Linear(self.d_model, self.d_v*self.h, bias=False)
    self.w_o = nn.Linear(self.d_v*self.h, self.d_model, bias=False)
    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
    d_k = query.shape[-1]
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attention_scores.masked_fill_(mask == 0, -1e9)
    attention_scores = attention_scores.softmax(dim=-1)
    if dropout is not None:
        attention_scores = dropout(attention_scores)
    return (attention_scores @ value), attention_scores

  def forward(self, q, k, v, mask):
    query = self.w_q(q)
    key = self.w_k(k)
    value = self.w_v(v)
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_v).transpose(1, 2)
    x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1 , self.h * self.d_v)
    return self.w_o(x)

class RMSNorm(nn.Module):
  def __init__(self,d_model:int,eps:float=1e-6):
    super().__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(2*d_model))

  def forward(self,x):
    return x*torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)*self.gamma

class FirstSection(nn.Module):
  def __init__(self,d_model:int,d_inner:int,d_conv:int,dropout:float):
    super().__init__()
    self.linear = nn.Linear(d_model,d_inner)
    self.conv1d = nn.Conv1d(d_inner,d_inner,d_conv,dilation=2,groups=d_inner,padding="same")
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    x = self.linear(x)
    x = x.transpose(1,2).contiguous()
    x = self.dropout(self.conv1d(x))
    x = x.transpose(1,2).contiguous()
    return x

class SecondSection(nn.Module):
  def __init__(self,d_model:int,d_inner:int,dropout:float):
    super().__init__()
    self.linear = nn.Linear(d_model,d_inner)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    return self.dropout(torch.relu(self.linear(x)))

class InsideProjectionLayer(nn.Module):
  def __init__(self,d_inner:int,d_model:int,dropout:float):
    super().__init__()
    self.linear = nn.Linear(d_inner,2*d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    return self.dropout(self.linear(x))

class SAJBlock(nn.Module):
  def __init__(self,d_model:int,first:FirstSection,self_attention_block:MultiHeadAttentionBlock,second:SecondSection,insideprojectionlayer:InsideProjectionLayer):
    super().__init__()
    self.first = first
    self.self_attention_block = self_attention_block
    self.second = second
    self.rms_norm = RMSNorm(d_model)
    self.insideprojectionlayer = insideprojectionlayer

  def forward(self,x,mask):
    to_add = x
    x = self.rms_norm(x)
    out = self.self_attention_block(x,x,x,mask)
    # out = self.rms_norm(out)
    first_out, second_out = out[:,:,:out.size(2)//2], out[:,:,out.size(2)//2:]
    first_out = self.first(first_out)
    second_out = self.second(second_out)
    out = first_out*second_out
    output = to_add + self.insideprojectionlayer(out)
    return output

class SAJLoop(nn.Module):
  def __init__(self,layers:nn.ModuleList):
    super().__init__()
    self.layers = layers

  def forward(self,x,device):
    mask = torch.tril(torch.ones(x.size(1),x.size(1),device=device))
    # mask = torch.eye(x.size(1),device=device)
    for layer in self.layers:
      x = layer(x,mask)

    return x

class FirstRepresentation(nn.Module):
  def __init__(self,d_model:int,d_features:int,dropout:float):
    super().__init__()
    self.rmsnorm = RMSNorm(d_model)
    self.linear = nn.Linear(2*d_model,d_features//2)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    return self.dropout(self.linear(self.rmsnorm(x)))

class AttentionCombine(nn.Module):
  def __init__(self,seq_len:int,d_features:int,dropout:float):
    super().__init__()
    self.linear = nn.Linear(seq_len,d_features//2)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    return self.dropout(torch.sigmoid(self.linear(x)))



class SAJ(nn.Module):
  def __init__(self,d_features:int,input_embedding:InputEmbedding,pos_encoding:PositionalEncoding,sajloop:SAJLoop,firstrepresentation:FirstRepresentation,attncom:AttentionCombine):
    super().__init__()
    self.input_embedding = input_embedding
    self.pos_encoding = pos_encoding
    self.sajloop = sajloop
    self.firstrepresentation = firstrepresentation
    self.linear = nn.Linear(d_features//2,d_features//2)
    self.attncom = attncom

  def firstrep(self,x,mask,device):
    x = torch.concat([x,mask],dim=-1)
    x = self.input_embedding(x)
    x = self.pos_encoding(x)
    x = self.sajloop(x,device)
    return self.firstrepresentation(x)

  def tilde_X1(self,x,mask,device):
    x = self.firstrep(x,mask,device)
    return self.linear(x)

  def weighted_combination_block(self,attn):
    attn = attn.squeeze(dim=1)
    if len(attn.shape)==4:
      attn = torch.transpose(attn,1,3)
      attn = attn.mean(dim=3)
      attn = torch.transpose(attn,1,2)
    eta = self.attncom(attn)
    return eta

  def compile(self,eta,first_output):
    return self.linear(eta*first_output)

def build_SAJ(seq_len:int,features:int,d_model:int=256,d_v:int=256,d_inner:int=128,d_conv:int=4,h:int=8,N:int=2,dropout:float=0.1,conv_dropout:float = 0., attn_dropout:float = 0.)->SAJ:
  features = 2*features
  embedded_input = InputEmbedding(features,d_model)
  positional_encoding = PositionalEncoding(d_model,seq_len,dropout)
  saj_blocks = []
  for _ in range(N):
    first = FirstSection(d_model,d_inner,d_conv,conv_dropout)
    self_attention_block = MultiHeadAttentionBlock(d_model,d_v,h,attn_dropout)
    second = SecondSection(d_model,d_inner,dropout)
    insideprojectionlayer = InsideProjectionLayer(d_inner,d_model,dropout)
    saj_block = SAJBlock(d_model,first,self_attention_block,second,insideprojectionlayer)
    saj_blocks.append(saj_block)

  sajloop = SAJLoop(nn.ModuleList(saj_blocks))
  firstrepresentation = FirstRepresentation(d_model,features,dropout)
  attncom = AttentionCombine(seq_len,features,dropout)

  saj = SAJ(features,embedded_input,positional_encoding,sajloop,firstrepresentation,attncom)

  return saj

