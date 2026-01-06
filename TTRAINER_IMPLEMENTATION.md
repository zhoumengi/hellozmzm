# TTrainer Implementation Documentation

## 概述 (Overview)

TTrainer 是一个继承自 TryTrainer 的高级训练器，专门为处理类别体素差异巨大的数据集设计。它保持了 TryTrainer 的 hybrid 注意力机制和损失函数（用于消融实验），同时添加了创新的层次化 Mamba 架构。

## 设计理念 (Design Philosophy)

### 1. 继承性 (Inheritance)
- **完全继承** TryTrainer 的所有功能
- **保持一致** hybrid 注意力机制（深层用类平衡注意力，浅层用多尺度注意力）
- **保持一致** Dice+CE 损失函数（Dice 项排除背景）

### 2. 创新点 (Innovations)

#### 编码器创新

##### 浅层（第1-2阶段）：局部感知Mamba
**目的**：在保留线性复杂度的同时，强制模型先建立局部范围内的强关联

**实现**：
- `WindowPartition3D`: 将特征图在空间上划分为重叠的局部窗口
- `LocalMamba3D`: 在每个窗口内独立应用 Mamba
- `WindowMerge3D`: 合并窗口结果，处理重叠区域

**关键参数**：
- `window_size`: (4, 4, 4) - 窗口大小
- `overlap_ratio`: 0.5 - 重叠比例

##### 中层（第3阶段）：层次化Mamba
**目的**：同时捕获器官级关系和结构内关系

**实现**：
- **路径A**（全局Mamba）：
  - 下采样特征（2x）
  - 全局 Mamba 处理（捕获器官级关系）
  - 上采样回原尺寸
  
- **路径B**（局部窗口Mamba）：
  - 使用 LocalMamba3D 处理原始分辨率特征
  - 捕获结构内关系

- **融合**：通过 1x1 卷积融合两个路径

##### 深层（瓶颈）：条件选择性Mamba
**目的**：让 Mamba 的选择性扫描机制参数由类别先验动态生成

**实现**：
1. **条件网络**：轻量子网络处理类别先验统计
2. **参数生成器**：生成 Mamba 的选择性参数（dt_scale, dt_shift等）
3. **条件调制**：通过缩放和偏移调制 Mamba 输入

**输入**：
- `x`: 特征图 (B, C, D, H, W)
- `class_priors`: 类别先验统计 (B, num_classes)

#### 解码器创新

##### 语义引导的Mamba细化模块（SG-Mamba）
**目的**：使用全局类别信息指导上采样恢复过程

**输入**：
- `f_up`: 上采样后的特征图
- `f_skip`: 来自跳跃连接的特征
- `global_class_vec`: 来自瓶颈层的全局类别向量

**过程**：
1. **特征融合**：融合 f_up 和 f_skip
2. **类别引导映射**：将全局类别向量映射为门控信号
3. **门控调制**：应用类别引导信号到融合特征
4. **Mamba细化**：使用 Mamba 处理门控特征
5. **残差连接**：保持信息流动
6. **输出投影**：投影到目标通道数

## 模块详解 (Module Details)

### WindowPartition3D
```python
class WindowPartition3D(nn.Module):
    """3D窗口划分模块"""
    def __init__(self, window_size=(4, 4, 4), overlap_ratio=0.5)
```
- **功能**：将3D特征图划分为重叠窗口
- **输入**：(B, C, D, H, W)
- **输出**：(B, num_windows, C, wd, wh, ww), (nD, nH, nW)

### WindowMerge3D
```python
class WindowMerge3D(nn.Module):
    """3D窗口合并模块"""
    def __init__(self, window_size=(4, 4, 4), overlap_ratio=0.5)
```
- **功能**：合并重叠窗口，平均重叠区域
- **输入**：windows, window_grid, original_shape
- **输出**：(B, C, D, H, W)

### LocalMamba3D
```python
class LocalMamba3D(nn.Module):
    """局部感知Mamba - 在窗口内独立应用Mamba"""
    def __init__(self, channels, window_size=(4, 4, 4), overlap_ratio=0.5,
                 d_state=16, d_conv=4, expand=2)
```
- **功能**：在每个局部窗口内独立应用 Mamba
- **降级方案**：如果 Mamba 不可用，使用深度可分离卷积
- **复杂度**：线性（相对于输入大小）

### HierarchicalMamba3D
```python
class HierarchicalMamba3D(nn.Module):
    """层次化Mamba - 并行全局和局部路径"""
    def __init__(self, channels, window_size=(4, 4, 4), 
                 downsample_factor=2, d_state=16, d_conv=4, expand=2)
```
- **路径A**：全局路径（下采样 -> Mamba -> 上采样）
- **路径B**：局部路径（LocalMamba3D）
- **融合**：拼接后1x1卷积

### ConditionalSelectiveMamba3D
```python
class ConditionalSelectiveMamba3D(nn.Module):
    """条件选择性Mamba - 参数由类别先验动态生成"""
    def __init__(self, channels, num_classes, d_state=16, d_conv=4, expand=2)
```
- **条件网络**：2层MLP (num_classes -> 128 -> 64)
- **参数生成**：生成 dt_scale, dt_shift, B_scale, C_scale
- **调制方式**：输入特征的缩放和偏移

### SemanticGuidedMambaRefinement
```python
class SemanticGuidedMambaRefinement(nn.Module):
    """语义引导的Mamba细化模块 - 用于解码器"""
    def __init__(self, up_channels, skip_channels, num_classes,
                 d_state=16, d_conv=4, expand=2)
```
- **类别引导映射器**：num_classes -> 128 -> fused_channels
- **门控机制**：Sigmoid 激活作为门控信号
- **Mamba细化**：处理门控后的特征
- **残差连接**：保持梯度流动

## TTrainer 类

### 初始化
```python
class TTrainer(TryTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device)
```

### 配置参数
```python
self.mamba_config = {
    'd_state': 16,          # Mamba 状态维度
    'd_conv': 4,            # Mamba 卷积核大小
    'expand': 2,            # Mamba 扩展因子
    'window_size': (4, 4, 4),       # 窗口大小
    'overlap_ratio': 0.5,           # 窗口重叠比例
    'downsample_factor': 2          # 下采样因子
}
```

### 关键方法

#### initialize_network()
重写网络初始化，在调用父类后添加层次化 Mamba 模块

#### _add_hierarchical_mamba_to_network()
核心方法，负责：
1. 为浅层（阶段0-1）添加 LocalMamba3D
2. 为中层（阶段2）添加 HierarchicalMamba3D
3. 为瓶颈层添加 ConditionalSelectiveMamba3D
4. 为解码器各阶段添加 SemanticGuidedMambaRefinement

#### _compute_class_priors(batch_data)
从批次数据计算类别先验统计
- 统计每个类别在批次中的体素比例
- 返回 (B, num_classes) 张量

#### _extract_global_class_vector(bottleneck_features)
从瓶颈特征提取全局类别向量
- 全局平均池化
- 线性投影到类别空间
- Softmax 归一化

## 使用方法 (Usage)

### 训练
```python
from LogWeightednnUNetTrainer import TTrainer

trainer = TTrainer(
    plans=plans,
    configuration=configuration,
    fold=fold,
    dataset_json=dataset_json,
    device=device
)

# 网络会自动初始化，包括所有 Mamba 模块
trainer.run_training()
```

### 消融实验
TTrainer 和 TryTrainer 的对比：
- **相同**：hybrid 注意力、Dice+CE 损失、数据增强策略
- **不同**：TTrainer 添加了层次化 Mamba 架构

这种设计使得可以直接比较 Mamba 创新的效果。

## 降级策略 (Fallback Strategy)

如果 `mamba-ssm` 不可用：
- **LocalMamba3D**：使用深度可分离卷积
- **HierarchicalMamba3D**：全局路径使用标准卷积
- **ConditionalSelectiveMamba3D**：使用 1x1 卷积 + BN + SiLU
- **SemanticGuidedMambaRefinement**：使用 3x3 卷积

## 性能考虑 (Performance Considerations)

### 内存使用
- **窗口划分**：增加了临时内存开销（窗口张量）
- **双路径**：HierarchicalMamba3D 需要额外内存存储两个路径
- **建议**：调整 `window_size` 和 `overlap_ratio` 以平衡性能

### 计算复杂度
- **LocalMamba3D**：O(N × w³) 其中 w 是窗口大小
- **HierarchicalMamba3D**：O(N/4 + N × w³) 全局+局部
- **条件网络**：可忽略不计

### 优化建议
1. 根据数据集大小调整 `window_size`
2. 对于小数据集，可以减少 `overlap_ratio`
3. 考虑使用混合精度训练（AMP）

## 与 TryTrainer 的区别总结

| 特性 | TryTrainer | TTrainer |
|------|-----------|----------|
| 基类 | nnUNetTrainer | TryTrainer |
| 注意力机制 | Hybrid（类平衡+多尺度） | ✅ 继承相同 |
| 损失函数 | Dice+CE（Dice排除背景） | ✅ 继承相同 |
| 编码器浅层 | 标准卷积 | ➕ 局部感知Mamba |
| 编码器中层 | 标准卷积 | ➕ 层次化Mamba |
| 编码器深层 | 标准卷积 | ➕ 条件选择性Mamba |
| 解码器 | 标准上采样+跳跃 | ➕ SG-Mamba细化 |

## 引用 (Citation)

如果使用此实现，请引用：
```
TTrainer: 层次化Mamba架构用于医学图像分割
- 局部感知Mamba（浅层）
- 层次化Mamba（中层）
- 条件选择性Mamba（深层）
- 语义引导Mamba细化（解码器）
```

## 问题排查 (Troubleshooting)

### 1. Mamba 不可用
**症状**：警告 "⚠️ mamba_ssm 不可用"
**解决**：
```bash
pip install mamba-ssm
```
或者：使用降级方案（自动启用）

### 2. 内存不足
**症状**：CUDA out of memory
**解决**：
- 减小 `window_size`
- 减小 `overlap_ratio`
- 减小批次大小
- 使用梯度累积

### 3. 注意力模块错误
**症状**：无法导入 attention_gates
**解决**：确保 `attention_gates (1).py` 在同一目录

## 未来改进 (Future Improvements)

1. **完全集成 Mamba**：目前的前向传播集成是简化版本，需要根据具体网络结构完善
2. **自适应窗口大小**：根据特征图尺寸动态调整窗口大小
3. **多尺度 Mamba**：在不同尺度应用 Mamba
4. **更复杂的条件网络**：使用更强的条件生成网络
5. **注意力-Mamba 融合**：探索注意力和 Mamba 的协同效应

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
