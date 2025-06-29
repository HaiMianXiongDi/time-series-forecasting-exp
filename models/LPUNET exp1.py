import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from layers.RevIN import RevIN
import logging
from scipy.fftpack import fft, ifft
from torch.nn.utils import weight_norm



# 配置日志记录
logging.basicConfig(filename='model_debug.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')





class UpDownSamplingLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UpDownSamplingLayer, self).__init__()
        self.layer = nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            nn.GELU(),
            torch.nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.layer(x)
    
    
    
class Predictor_test(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predictor_test, self).__init__()
        self.layer = nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            nn.GELU(),
            torch.nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.layer(x)
    


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.input_channels = configs.enc_in
        self.original_input_len = configs.seq_len
        self.out_len = configs.pred_len
        self.individual = configs.individual
        self.stage_num = configs.stage_num
        self.stage_pool_kernel = configs.stage_pool_kernel
        self.stage_pool_stride = configs.stage_pool_stride
        self.stage_pool_padding = configs.stage_pool_padding   
        self.freq_downsampling_percentage = configs.freq_downsampling_percentage

        self.revin_layer = RevIN(self.input_channels, affine=True, subtract_last=False)

        # ✨插值倍率：例如 1.5 表示降采样为原始 1/1.5，再上采样回去
        self.interp_ratio = getattr(configs, 'interp_ratios', 1.0)

        self.input_len = self.original_input_len  # 预测器仍然接收原始长度
        self.predictor = Predictor_test(self.input_len, self.out_len)

    def forward(self, x):
        # Step-0 归一化 & 维度整理
        x_norm = self.revin_layer(x, 'norm')        # [B, L, C]
        seq_level = x_norm.permute(0, 2, 1)         # [B, C, L]
        logging.info(f"Input after permute: {seq_level.shape}")

        # Step-1 插值降采样 & 恢复（池化行为）
        if self.interp_ratio != 1.0:
            down_len = max(1, int(self.original_input_len / self.interp_ratio))
            logging.info(f"Interpolating down to {down_len} steps, then back to {self.original_input_len}")
            seq_level = F.interpolate(seq_level, size=down_len, mode='linear', align_corners=False)
            seq_level = F.interpolate(seq_level, size=self.original_input_len, mode='linear', align_corners=False)

        # Step-2 预测
        final_prediction = self.predictor(seq_level)         # [B, C, out_len]
        final_prediction = final_prediction.permute(0, 2, 1) # [B, out_len, C]

        # Step-3 反归一化
        final_prediction = self.revin_layer(final_prediction, 'denorm')
        return final_prediction