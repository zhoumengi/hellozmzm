# TTrainer 2D Mamba Support

## 更新说明

TTrainer 现在完全支持 2D 和 3D 配置，并为每种模式提供专门优化的 Mamba 模块。

## 新增功能

### 1. 2D Mamba 模块

为 2D 配置创建了专门的 Mamba 模块，处理 `(B, C, H, W)` 张量：

- **WindowPartition2D** - 2D 窗口划分（窗口大小：8×8）
- **WindowMerge2D** - 2D 窗口合并
- **LocalMamba2D** - 2D 局部感知 Mamba
- **HierarchicalMamba2D** - 2D 层次化 Mamba（全局+局部双路径）
- **ConditionalSelectiveMamba2D** - 2D 条件选择性 Mamba
- **SemanticGuidedMambaRefinement2D** - 2D 语义引导 Mamba 细化

### 2. 自动检测与适配

TTrainer 会自动检测配置类型并选择合适的模块：

```python
# 检测2D/3D
if '2d' in configuration_name:
    使用 2D Mamba 模块
else:
    使用 3D Mamba 模块
```

### 3. 详细的输出提示

初始化时会显示完整的配置信息：

```
================================================================================
🔥 初始化TTrainer - 层次化Mamba创新训练器
================================================================================
✅ 继承TryTrainer的hybrid注意力和损失函数
🎯 添加层次化Mamba创新:
   - 浅层(1-2): 局部感知Mamba
   - 中层(3): 层次化Mamba
   - 深层(瓶颈): 条件选择性Mamba
   - 解码器: 语义引导Mamba细化
================================================================================

✅ 检测到2D配置，将使用2D Mamba模块

📊 从configuration提取通道信息:
  编码器通道: [32, 64, 128, 256, 512, 512, 512]
  解码器通道: [512, 512, 512, 256, 128, 64, 32]

🚀 准备添加Mamba模块 (2D模式)...

================================================================================
🔥 添加层次化Mamba模块到网络 (2D模式)...
================================================================================

📍 阶段 1: 添加局部感知Mamba2D (channels=32)
  ✅ 局部感知Mamba2D已添加

📍 阶段 2: 添加局部感知Mamba2D (channels=64)
  ✅ 局部感知Mamba2D已添加

📍 阶段 3: 添加层次化Mamba2D (channels=128)
  ✅ 层次化Mamba2D已添加

📍 瓶颈层: 添加条件选择性Mamba2D (channels=512)
  ✅ 条件选择性Mamba2D已添加

📍 解码器: 添加语义引导Mamba2D细化模块
  ✅ 解码器阶段 0: SG-Mamba2D (up=512, skip=512)
  ✅ 解码器阶段 1: SG-Mamba2D (up=512, skip=512)
  ✅ 解码器阶段 2: SG-Mamba2D (up=512, skip=256)
  ✅ 解码器阶段 3: SG-Mamba2D (up=256, skip=128)
  ✅ 解码器阶段 4: SG-Mamba2D (up=128, skip=64)
  ✅ 解码器阶段 5: SG-Mamba2D (up=64, skip=32)

🎉 层次化Mamba2D模块添加完成!
  配置维度: 2D (H, W)
  窗口大小: (8, 8)
  总Mamba参数: 12,345,678
  Mamba可用性: ✅ 使用mamba-ssm
================================================================================

✅ Mamba模块成功集成到网络中!
```

## 配置参数

### 2D 配置
```python
mamba_config = {
    'd_state': 16,
    'd_conv': 4,
    'expand': 2,
    'window_size': (8, 8),      # 2D窗口大小
    'overlap_ratio': 0.5,
    'downsample_factor': 2,
    'use_mamba': True           # 启用Mamba
}
```

### 3D 配置
```python
mamba_config = {
    'd_state': 16,
    'd_conv': 4,
    'expand': 2,
    'window_size': (4, 4, 4),   # 3D窗口大小
    'overlap_ratio': 0.5,
    'downsample_factor': 2,
    'use_mamba': True           # 启用Mamba
}
```

## 使用方法

### 训练 2D 模型
```bash
nnUNetv2_train DATASET_ID 2d FOLD -tr TTrainer
```

### 训练 3D 模型
```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr TTrainer
```

## 技术细节

### 2D vs 3D 差异

| 特性 | 2D | 3D |
|------|----|----|
| 输入形状 | (B, C, H, W) | (B, C, D, H, W) |
| 窗口大小 | (8, 8) | (4, 4, 4) |
| 卷积操作 | Conv2d | Conv3d |
| 批归一化 | BatchNorm2d | BatchNorm3d |
| 窗口维度 | 2D (H, W) | 3D (D, H, W) |

### 模块对应关系

| 3D 模块 | 2D 模块 |
|---------|---------|
| WindowPartition3D | WindowPartition2D |
| WindowMerge3D | WindowMerge2D |
| LocalMamba3D | LocalMamba2D |
| HierarchicalMamba3D | HierarchicalMamba2D |
| ConditionalSelectiveMamba3D | ConditionalSelectiveMamba2D |
| SemanticGuidedMambaRefinement | SemanticGuidedMambaRefinement2D |

### 降级策略

如果 `mamba-ssm` 不可用，所有模块会自动降级到卷积实现：

- **2D**: 使用 `Conv2d`, `BatchNorm2d`
- **3D**: 使用 `Conv3d`, `BatchNorm3d`

## 验证

### 检查点 1: 初始化消息
启动训练时，应该看到：
- ✅ "检测到2D配置，将使用2D Mamba模块"（2D模式）
- ✅ "添加层次化Mamba模块到网络 (2D模式)..." 或 "(3D模式)..."

### 检查点 2: 模块添加
应该看到每个阶段的成功消息：
- ✅ "局部感知Mamba2D已添加"（2D）或 "局部感知Mamba3D已添加"（3D）
- ✅ "层次化Mamba2D已添加"（2D）或 "层次化Mamba3D已添加"（3D）
- ✅ 等等...

### 检查点 3: 总结信息
最后应该看到：
- ✅ "层次化Mamba2D模块添加完成!"（2D）或 "层次化Mamba3D模块添加完成!"（3D）
- ✅ 配置维度: 2D (H, W) 或 3D (D, H, W)
- ✅ 窗口大小
- ✅ 总参数数量
- ✅ Mamba可用性状态

### 检查点 4: 最终确认
- ✅ "Mamba模块成功集成到网络中!"

## 性能考虑

### 2D 优势
- ✅ 内存使用更少
- ✅ 训练速度更快
- ✅ 更大的窗口大小 (8×8 vs 4×4×4)
- ✅ 适合高分辨率 2D 医学图像

### 3D 优势
- ✅ 完整的体素关联
- ✅ 更强的空间上下文
- ✅ 适合 CT/MRI 等 3D 数据

## 故障排除

### 问题: 没有看到 Mamba 添加消息
**原因**: 可能初始化失败
**解决**: 检查日志中的错误消息

### 问题: 显示"Mamba模块已禁用"
**原因**: 配置中 `use_mamba` 被设为 `False`
**解决**: 这是旧版本的行为，新版本 2D 也启用 Mamba

### 问题: 内存不足
**解决**: 
- 减小 `window_size`: 2D 从 (8,8) 改为 (4,4)
- 减小 `batch_size`
- 减小 `d_state` 或 `expand`

## 更新历史

- **2024-12-30**: 添加完整的 2D Mamba 支持
- **2024-12-30**: 添加详细的输出提示
- **2024-12-30**: 自动检测 2D/3D 配置

## 下一步

考虑的改进：
1. 可调节的窗口大小（根据输入分辨率自适应）
2. 混合 2D-3D Mamba（例如 2.5D）
3. 多尺度窗口策略
4. 动态窗口大小选择
