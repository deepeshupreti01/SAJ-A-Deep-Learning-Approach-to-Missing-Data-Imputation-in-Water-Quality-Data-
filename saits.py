import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import masked_mae_cal

class SAITS(nn.Module):
    def __init__(
        self,
        n_groups,
        n_group_inner_layers,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        attn_dropout
    ):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = True
        actual_d_feature = d_feature * 2
        self.param_sharing_strategy = "inner_group"
        self.MIT = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layer_stack_for_first_block = nn.ModuleList(
            [
                EncoderLayer(
                    d_time,
                    actual_d_feature,
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
                )
                for _ in range(n_groups)
            ]
        )

        self.layer_stack_for_second_block = nn.ModuleList(
            [
                EncoderLayer(
                    d_time,
                    actual_d_feature,
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
                )
                for _ in range(n_groups)
            ]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        # for the 1st block
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for the 2nd block
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for the 3rd block
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]
        # the first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely term e in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # the second DMSA block
        input_X_for_second = (
            torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        )
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )  # namely term alpha in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # the attention-weighted combination block
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        # replace non-missing part with original data
        X_c = masks * X + (1 - masks) * X_tilde_3
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

def build_SAITS(n_groups,n_group_inner_layers,seq_len,feature_num,d_model,d_inner, h,d_k,d_v,dropout,attn_dropout)->SAITS:
  saits = SAITS(n_groups,n_group_inner_layers,seq_len,feature_num,d_model,d_inner,h,d_k,d_v,dropout,attn_dropout)
  return saits

