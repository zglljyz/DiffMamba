import math
import torch
from torch import nn
from mamba_ssm import Mamba


class DiffSpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True):
        super(DiffSpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.gating_linear = nn.Linear(channels, channels)

        self.act_sigmoid = nn.Sigmoid()
        
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )
        self.alpha = nn.Parameter(torch.tensor(0.50))

    def forward(self, x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        
        x_flat_2d = x_re.view(-1, C) 
        
        x_in_pool = x_flat_2d.unsqueeze(1)
        
        x_smooth = torch.nn.functional.avg_pool1d(
            x_in_pool, kernel_size=5, stride=1, padding=2, count_include_pad=False
        )
        
        x_diff = x_in_pool - self.alpha*x_smooth

        x_mamba_in = x_diff.view(1, -1, C)
        
        mamba_out = self.mamba(x_mamba_in)
        
        gate_score = self.act_sigmoid(self.gating_linear(x_mamba_in))
        x_gated = mamba_out * gate_score

        x_recon = x_gated.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        
        if self.use_proj:
            x_recon = self.proj(x_recon)
            
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class SpaMamba(nn.Module):
    def __init__(self,channels,use_residual=True,group_num=4,use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  
                           d_model=channels,  
                           d_state=16,  
                           d_conv=4,  
                           expand=2, 
                           )
        
        self.gating_linear = nn.Linear(channels, channels)
        self.act_sigmoid = nn.Sigmoid()

        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self,x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C = x_re.shape

        x_flat = x_re.view(1,-1, C)
        
        mamba_out = self.mamba(x_flat)
        
        gate_score = self.act_sigmoid(self.gating_linear(x_flat))
        
        x_gated = mamba_out * gate_score

        x_recon = x_gated.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self,channels,token_num,use_residual,group_num=4,use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels,use_residual=use_residual,group_num=group_num)
        self.diff_mamba = DiffSpaMamba(channels, use_residual=use_residual, group_num=group_num)
    def forward(self,x):
        out_spa = self.spa_mamba(x)
        
        out_diff = self.diff_mamba(x)

        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = out_spa * weights[0] + out_diff * weights[1]
        else:
            fusion_x = out_spa + out_diff

        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class DiffMamba(nn.Module):
    def __init__(self,in_channels=128,hidden_dim=64,num_classes=10,use_residual=True,mamba_type='both',token_num=4,group_num=4,use_att=True):
        super(DiffMamba, self).__init__()
        self.mamba_type = mamba_type
        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0),
                                             nn.GroupNorm(group_num,hidden_dim),
                                             nn.SiLU())
        
        self.downsample = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                                        nn.MaxPool2d(kernel_size=2), 
                                        nn.GELU())
        if mamba_type == 'spa':
            self.mamba = nn.Sequential(
                                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                       SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                      )
            
        elif mamba_type == 'diff':
            self.mamba = nn.Sequential(
                                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                       DiffSpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                      )
    
        elif mamba_type=='both':
            self.mamba = nn.Sequential(
                                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                       BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       )

        self.cls_head = nn.Sequential(nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
                                      nn.GroupNorm(group_num,128),
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels=128,out_channels=num_classes,kernel_size=1,stride=1,padding=0))

    def forward(self,x):
        x = self.patch_embedding(x)

        x = self.downsample(x)

        x = self.mamba(x)

        logits = self.cls_head(x)

        return logits
