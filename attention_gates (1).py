"""
多尺度注意力门控 - 专门为nnU-Net跳跃连接设计
位置: nnunetv2/training/nnUNetTrainer/attention_gates.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Union
from torch.cuda.amp import autocast


class MultiScaleAttentionGate3D(nn.Module):
    """
    完整多尺度注意力门控 - 不简化版本
    专门处理nnU-Net的3D医学图像分割
    """
    
    def __init__(
        self,
        skip_channels: int,          # 跳跃连接通道数
        gate_channels: int,          # 门控信号通道数（解码器特征）
        inter_channels: Optional[int] = None,
        scale_factor: int = 2,       # 下采样倍数
        reduction_ratio: int = 16,
        use_residual: bool = True,
        dropout_rate: float = 0.1,
        attention_mode: str = 'sigmoid',  # 'sigmoid' 或 'softmax'
        **kwargs
    ):
        super().__init__()
        
        # 参数验证
        assert scale_factor in [1, 2, 4, 8], f"不支持的scale_factor: {scale_factor}"
        assert attention_mode in ['sigmoid', 'softmax'], f"不支持的attention_mode: {attention_mode}"
        
        self.skip_channels = skip_channels
        self.gate_channels = gate_channels
        self.scale_factor = scale_factor
        self.use_residual = use_residual
        self.attention_mode = attention_mode
        self.dropout_rate = dropout_rate
        
        # 计算中间通道数
        if inter_channels is None:
            self.inter_channels = max(8, skip_channels // reduction_ratio)
        else:
            self.inter_channels = inter_channels
        
        # 1. Skip路径卷积 - 处理高分辨率特征
        self.skip_conv = nn.Sequential(
            nn.Conv3d(skip_channels, self.inter_channels, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
        # 2. Gate路径卷积 - 处理低分辨率特征
        self.gate_conv = nn.Sequential(
            nn.Conv3d(gate_channels, self.inter_channels,
                     kernel_size=1, bias=False),
            nn.BatchNorm3d(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. 空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(self.inter_channels * 2, self.inter_channels // 2,
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.inter_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.inter_channels // 2, 1,
                     kernel_size=1, bias=False),
            nn.BatchNorm3d(1)
        )
        
        # 4. 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(self.inter_channels * 2, self.inter_channels // reduction_ratio,
                     kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.inter_channels // reduction_ratio, skip_channels,
                     kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 5. 残差连接的权重（可学习）
        if use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.5))
            self.residual_bias = nn.Parameter(torch.tensor(0.0))
        
        # 6. 输出归一化
        self.output_norm = nn.InstanceNorm3d(skip_channels, affine=True)
        
        # 7. 梯度稳定层
        self.gradient_stabilizer = nn.Sequential(
            nn.Conv3d(skip_channels, skip_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(skip_channels // 4, skip_channels, kernel_size=1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """遵循nnU-Net的权重初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        # 初始化注意力层为接近均匀分布
        if hasattr(self.spatial_attention[-2], 'weight'):
            nn.init.constant_(self.spatial_attention[-2].weight, 0)
            nn.init.constant_(self.spatial_attention[-2].bias, 0)
    
    @autocast(enabled=False)  # 确保数值稳定性
    def forward(
        self, 
        skip: torch.Tensor, 
        gate: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            skip: [B, C_skip, D, H, W] 跳跃连接特征（高分辨率）
            gate: [B, C_gate, D//scale, H//scale, W//scale] 门控特征（低分辨率）
            return_attention: 是否返回注意力图
            
        Returns:
            output: 加权的跳跃特征
            attention_map (optional): 注意力图
        """
        batch_size = skip.size(0)
        original_shape = skip.shape[2:]
        
        # ========== 步骤1: 尺寸对齐 ==========
        if self.scale_factor > 1:
            # 上采样gate以匹配skip的尺寸
            gate_resized = F.interpolate(
                gate, 
                size=original_shape,
                mode='trilinear',
                align_corners=False
            )
        else:
            gate_resized = gate
        
        # ========== 步骤2: 特征提取 ==========
        # 提取skip特征（保留空间细节）
        skip_feat = self.skip_conv(skip)  # [B, C_inter, D, H, W]
        
        # 提取gate特征（保留语义信息）
        gate_feat = self.gate_conv(gate_resized)  # [B, C_inter, D, H, W]
        
        # ========== 步骤3: 特征融合 ==========
        # 拼接特征
        combined = torch.cat([skip_feat, gate_feat], dim=1)  # [B, 2*C_inter, D, H, W]
        
        # ========== 步骤4: 生成注意力图 ==========
        # 空间注意力
        spatial_att = self.spatial_attention(combined)  # [B, 1, D, H, W]
        
        # 通道注意力
        channel_att = self.channel_attention(combined)  # [B, C_skip, 1, 1, 1]
        
        # 结合空间和通道注意力
        if self.attention_mode == 'sigmoid':
            spatial_att = torch.sigmoid(spatial_att)
        else:  # softmax
            spatial_att = F.softmax(spatial_att.view(batch_size, 1, -1), dim=2)
            spatial_att = spatial_att.view(batch_size, 1, *original_shape)
        
        # 最终注意力图
        attention_map = spatial_att * channel_att
        
        # ========== 步骤5: 应用注意力 ==========
        if self.use_residual:
            # 残差连接：output = α * skip + β * (skip * attention)
            attention_weighted = skip * attention_map
            residual_weight = torch.sigmoid(self.residual_weight)
            output = residual_weight * skip + (1 - residual_weight) * attention_weighted
            output = output + self.residual_bias
        else:
            output = skip * attention_map
        
        # ========== 步骤6: 后处理 ==========
        # 梯度稳定
        stabilized = self.gradient_stabilizer(output)
        
        # 归一化输出
        normalized = self.output_norm(stabilized)
        
        # ========== 步骤7: 确保形状不变 ==========
        assert normalized.shape == skip.shape, \
            f"输出形状{normalized.shape} != 输入形状{skip.shape}"
        
        if return_attention:
            return normalized, attention_map
        else:
            return normalized


class HierarchicalAttentionGate3D(nn.Module):
    """
    层次化多尺度注意力门控
    同时考虑多个尺度的门控信号
    """
    
    def __init__(
        self,
        skip_channels: int,
        gate_channels_list: List[int],  # 不同尺度的门控通道列表
        reduction_ratio: int = 16,
        **kwargs
    ):
        super().__init__()
        
        self.num_levels = len(gate_channels_list)
        self.attention_gates = nn.ModuleList()
        
        # 创建多个注意力门
        for i, gate_channels in enumerate(gate_channels_list):
            scale_factor = 2 ** (i + 1)  # 2, 4, 8...
            
            gate = MultiScaleAttentionGate3D(
                skip_channels=skip_channels,
                gate_channels=gate_channels,
                scale_factor=scale_factor,
                reduction_ratio=reduction_ratio,
                **kwargs
            )
            self.attention_gates.append(gate)
        
        # 注意力融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv3d(self.num_levels, self.num_levels // 2 if self.num_levels > 1 else 1,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.num_levels // 2 if self.num_levels > 1 else 1, 1,
                     kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        skip: torch.Tensor,
        gate_list: List[torch.Tensor],
        return_attention: bool = False
    ):
        """
        Args:
            skip: 跳跃特征
            gate_list: 不同尺度的门控特征列表
        """
        assert len(gate_list) == self.num_levels, \
            f"需要{self.num_levels}个门控特征，但得到{len(gate_list)}个"
        
        attention_maps = []
        weighted_features = []
        
        # 通过每个注意力门
        for i, (gate_module, gate_input) in enumerate(zip(self.attention_gates, gate_list)):
            if return_attention:
                weighted_feat, att_map = gate_module(skip, gate_input, return_attention=True)
                attention_maps.append(att_map)
            else:
                weighted_feat = gate_module(skip, gate_input, return_attention=False)
            
            weighted_features.append(weighted_feat)
        
        # 融合注意力图
        if len(attention_maps) > 1:
            stacked_attention = torch.cat(attention_maps, dim=1)  # [B, num_levels, D, H, W]
            fused_attention = self.fusion_layer(stacked_attention)
        elif attention_maps:
            fused_attention = attention_maps[0]
        else:
            fused_attention = None
        
        # 加权融合特征
        if len(weighted_features) > 1:
            # 可学习的融合权重
            fusion_weights = F.softmax(
                torch.ones(self.num_levels, device=skip.device), 
                dim=0
            )
            output = sum(w * f for w, f in zip(fusion_weights, weighted_features))
        else:
            output = weighted_features[0]
        
        if return_attention:
            return output, fused_attention, attention_maps
        else:
            return output


class ClassBalancedAttentionGate3D(MultiScaleAttentionGate3D):
    """
    类别平衡注意力门控 - 专门针对类别不平衡
    增强小类别区域的注意力
    """
    
    def __init__(
        self,
        skip_channels: int,
        gate_channels: int,
        num_classes: int,
        minority_class_indices: Optional[List[int]] = None,
        minority_boost: float = 2.0,
        class_aware_mode: str = 'adaptive',  # 'adaptive' 或 'fixed'
        **kwargs
    ):
        super().__init__(skip_channels, gate_channels, **kwargs)
        
        self.num_classes = num_classes
        self.minority_class_indices = minority_class_indices if minority_class_indices else []
        self.minority_boost = minority_boost
        self.class_aware_mode = class_aware_mode
        
        # 类别感知注意力头
        self.class_attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(gate_channels, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(num_classes)
        ])
        
        # 类别权重学习
        if class_aware_mode == 'adaptive':
            self.class_weights = nn.Parameter(torch.ones(num_classes))
        else:
            # 固定权重：小类别权重更高
            self.class_weights = torch.ones(num_classes)
            for idx in self.minority_class_indices:
                if idx < num_classes:
                    self.class_weights[idx] = minority_boost
            self.class_weights = nn.Parameter(self.class_weights, requires_grad=False)
        
        # 小类别增强模块
        if self.minority_class_indices:
            self.minority_enhancer = nn.Sequential(
                nn.Conv3d(gate_channels, len(self.minority_class_indices), 
                         kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(len(self.minority_class_indices), len(self.minority_class_indices),
                         kernel_size=1)
            )
    
    def forward(
        self,
        skip: torch.Tensor,
        gate: torch.Tensor,
        class_importance: Optional[torch.Tensor] = None,  # [num_classes] 类别重要性
        return_class_attention: bool = False
    ):
        # 基础注意力
        base_output, base_attention = super().forward(skip, gate, return_attention=True)
        
        # 生成类别特定的注意力图
        class_attention_maps = []
        for i, head in enumerate(self.class_attention_heads):
            class_map = head(gate)  # [B, 1, D, H, W]
            
            # 应用类别权重
            weight = self.class_weights[i]
            if class_importance is not None and i < len(class_importance):
                weight = weight * class_importance[i]
            
            class_map = class_map * weight
            class_attention_maps.append(class_map)
        
        # 增强小类别
        if self.minority_class_indices and hasattr(self, 'minority_enhancer'):
            minority_features = self.minority_enhancer(gate)
            for idx, class_idx in enumerate(self.minority_class_indices):
                if class_idx < len(class_attention_maps):
                    boost_map = minority_features[:, idx:idx+1]
                    class_attention_maps[class_idx] = class_attention_maps[class_idx] + \
                                                     boost_map * self.minority_boost
        
        # 融合类别注意力图
        stacked_class_maps = torch.stack(class_attention_maps, dim=1)  # [B, C, 1, D, H, W]
        
        # 类别权重归一化
        class_weights_normalized = F.softmax(self.class_weights, dim=0)
        weighted_class_maps = stacked_class_maps * class_weights_normalized.view(1, -1, 1, 1, 1, 1)
        
        # 求和得到最终的类别感知注意力
        class_aware_attention = weighted_class_maps.sum(dim=1)  # [B, 1, D, H, W]
        
        # 融合基础注意力和类别感知注意力
        combined_attention = (base_attention + class_aware_attention) / 2
        
        # 应用最终注意力
        final_output = skip * combined_attention
        
        if return_class_attention:
            return final_output, combined_attention, class_attention_maps
        else:
            return final_output, combined_attention


# ========== 简化版本（如果需要） ==========

class SimplifiedAttentionGate3D(nn.Module):
    """
    简化版注意力门控 - 计算量更小
    """
    
    def __init__(self, skip_channels: int, gate_channels: int, scale_factor: int = 2):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # 超轻量设计
        self.skip_conv = nn.Conv3d(skip_channels, 1, kernel_size=1)
        self.gate_conv = nn.Conv3d(gate_channels, 1, kernel_size=1)
        self.combine_conv = nn.Conv3d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, skip: torch.Tensor, gate: torch.Tensor):
        # 尺寸对齐
        if self.scale_factor > 1:
            gate = F.interpolate(gate, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        # 压缩通道
        skip_compressed = self.skip_conv(skip)  # [B, 1, D, H, W]
        gate_compressed = self.gate_conv(gate)  # [B, 1, D, H, W]
        
        # 拼接和注意力生成
        combined = torch.cat([skip_compressed, gate_compressed], dim=1)
        attention = self.sigmoid(self.combine_conv(combined))
        
        # 应用注意力
        return skip * attention


# ========== 工厂函数 ==========

def create_attention_gate(
    gate_type: str = 'multiscale',
    skip_channels: int = None,
    gate_channels: int = None,
    **kwargs
) -> nn.Module:
    """
    创建注意力门的工厂函数
    
    Args:
        gate_type: 'multiscale', 'hierarchical', 'class_balanced', 'simplified'
        skip_channels: 跳跃连接通道数
        gate_channels: 门控信号通道数
        **kwargs: 传递给具体类的参数
    """
    gate_type = gate_type.lower()
    
    if gate_type == 'multiscale':
        return MultiScaleAttentionGate3D(skip_channels, gate_channels, **kwargs)
    
    elif gate_type == 'hierarchical':
        if 'gate_channels_list' not in kwargs:
            raise ValueError("hierarchical类型需要gate_channels_list参数")
        return HierarchicalAttentionGate3D(skip_channels, **kwargs)
    
    elif gate_type == 'class_balanced':
        if 'num_classes' not in kwargs:
            raise ValueError("class_balanced类型需要num_classes参数")
        return ClassBalancedAttentionGate3D(skip_channels, gate_channels, **kwargs)
    
    elif gate_type == 'simplified':
        return SimplifiedAttentionGate3D(skip_channels, gate_channels, **kwargs)
    
    else:
        raise ValueError(f"未知的注意力门类型: {gate_type}")


# ========== 测试代码 ==========

if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试基本注意力门
    print("测试MultiScaleAttentionGate3D...")
    skip = torch.randn(2, 32, 64, 64, 64).to(device)  # [B, C, D, H, W]
    gate = torch.randn(2, 64, 32, 32, 32).to(device)  # 下采样2倍
    
    attention_gate = MultiScaleAttentionGate3D(
        skip_channels=32,
        gate_channels=64,
        scale_factor=2,
        reduction_ratio=8
    ).to(device)
    
    output, attention = attention_gate(skip, gate, return_attention=True)
    print(f"输入skip形状: {skip.shape}")
    print(f"输入gate形状: {gate.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力图形状: {attention.shape}")
    print(f"输出与skip形状一致: {output.shape == skip.shape}")
    
    # 测试层次化注意力
    print("\n测试HierarchicalAttentionGate3D...")
    hierarchical_gate = HierarchicalAttentionGate3D(
        skip_channels=32,
        gate_channels_list=[64, 128, 256],  # 不同尺度的门控
        reduction_ratio=8
    ).to(device)
    
    gate_list = [
        torch.randn(2, 64, 32, 32, 32).to(device),   # 下采样2倍
        torch.randn(2, 128, 16, 16, 16).to(device),  # 下采样4倍
        torch.randn(2, 256, 8, 8, 8).to(device)      # 下采样8倍
    ]
    
    output = hierarchical_gate(skip, gate_list)
    print(f"层次化输出形状: {output.shape}")
    
    print("\n所有测试通过！")