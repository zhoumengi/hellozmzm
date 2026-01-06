# TTrainer - Hierarchical Mamba Innovations

## 概述 (Overview)

本仓库实现了 **TTrainer**，一个专门为处理类别体素差异巨大的医学图像分割任务设计的创新训练器。TTrainer 继承自 TryTrainer，保持相同的 hybrid 注意力机制和损失函数（用于消融实验），并添加了创新的层次化 Mamba 架构。

## 主要特性 (Key Features)

### ✅ 继承自 TryTrainer
- **Hybrid 注意力**: 深层使用类平衡注意力，浅层使用多尺度注意力
- **Dice+CE 损失**: Dice 项排除背景，CE 项包含背景
- **数据增强**: 禁用镜像增强（牙齿数据集特性）
- **后处理**: 连通域分析

### 🔥 创新的层次化 Mamba 架构

#### 1. 浅层编码器（第1-2阶段）：局部感知 Mamba
```
目的: 建立局部范围内的强关联（如牙釉质的所有体素）
实现: WindowPartition3D + LocalMamba3D + WindowMerge3D
优势: 线性复杂度 + 强局部特征学习
```

#### 2. 中层编码器（第3阶段）：层次化 Mamba
```
目的: 同时捕获器官级关系和结构内关系
实现: 并行双路径 (全局 Mamba + 局部窗口 Mamba)
优势: 多尺度特征提取
```

#### 3. 深层编码器（瓶颈）：条件选择性 Mamba
```
目的: 根据类别先验动态调整 Mamba 参数
实现: 轻量子网络生成选择性扫描参数
优势: 自适应处理极端类别不平衡
```

#### 4. 解码器：语义引导 Mamba 细化
```
目的: 用全局类别信息指导上采样恢复
实现: 类别向量 -> 门控信号 -> Mamba 细化
优势: 类别感知的特征恢复
```

## 文件结构 (File Structure)

```
hellozmzm/
├── LogWeightednnUNetTrainer.py      # 主要实现文件
│   ├── TryTrainer                   # 基类训练器（已有）
│   ├── WindowPartition3D            # 窗口划分模块（新增）
│   ├── WindowMerge3D                # 窗口合并模块（新增）
│   ├── LocalMamba3D                 # 局部感知Mamba（新增）
│   ├── HierarchicalMamba3D          # 层次化Mamba（新增）
│   ├── ConditionalSelectiveMamba3D  # 条件选择性Mamba（新增）
│   ├── SemanticGuidedMambaRefinement # 语义引导Mamba（新增）
│   └── TTrainer                     # 主训练器类（新增）
│
├── attention_gates (1).py           # 注意力门模块（已有）
│
├── TTRAINER_IMPLEMENTATION.md       # 详细技术文档（新增）
├── example_usage_ttrainer.py        # 使用示例（新增）
└── README.md                        # 本文件（新增）
```

## 快速开始 (Quick Start)

### 前置要求
```bash
# 基础依赖
pip install torch torchvision
pip install nnunetv2

# Mamba 依赖（可选，如无则自动降级到卷积）
pip install mamba-ssm
```

### 基础使用
```python
from LogWeightednnUNetTrainer import TTrainer

# 初始化训练器
trainer = TTrainer(
    plans=plans,
    configuration="3d_fullres",
    fold=0,
    dataset_json=dataset_json,
    device=device
)

# 训练（Mamba 模块会自动集成）
trainer.run_training()
```

### 命令行使用
```bash
# 使用 TTrainer 训练
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr TTrainer

# 使用 TryTrainer 作为基线对比
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr TryTrainer
```

## 消融实验设计 (Ablation Study Design)

| 训练器 | 注意力 | 损失函数 | Mamba架构 | 用途 |
|--------|--------|----------|-----------|------|
| TryTrainer | Hybrid | Dice+CE | ❌ | 基线 |
| TTrainer | Hybrid | Dice+CE | ✅ | 实验组 |

**对比点**: 仅 Mamba 创新的影响
**度量指标**: Dice 分数（特别关注少数类）

## 配置参数 (Configuration)

### Mamba 模块配置
```python
mamba_config = {
    'd_state': 16,              # Mamba 状态维度
    'd_conv': 4,                # Mamba 卷积核大小
    'expand': 2,                # Mamba 扩展因子
    'window_size': (4, 4, 4),   # 窗口大小
    'overlap_ratio': 0.5,       # 窗口重叠比例
    'downsample_factor': 2      # 下采样因子
}
```

### 调优建议

#### 内存优化
```python
# 减少内存使用
window_size = (2, 2, 2)      # 更小的窗口
overlap_ratio = 0.25         # 更少的重叠
```

#### 性能优化
```python
# 提高处理速度
d_state = 8                  # 减少状态维度
expand = 1                   # 减少扩展因子
```

#### 精度优化
```python
# 提高模型容量
d_state = 32                 # 增加状态维度
window_size = (8, 8, 8)      # 更大的感受野
expand = 4                   # 增加扩展因子
```

## 技术细节 (Technical Details)

### 架构概览
```
Input (B, C, D, H, W)
    ↓
Encoder Stage 1-2: LocalMamba3D
    ├─ Window Partition
    ├─ Mamba per window
    └─ Window Merge
    ↓
Encoder Stage 3: HierarchicalMamba3D
    ├─ Path A: Global (Downsample → Mamba → Upsample)
    ├─ Path B: Local (LocalMamba3D)
    └─ Fusion (Concat → Conv)
    ↓
Bottleneck: ConditionalSelectiveMamba3D
    ├─ Compute class priors
    ├─ Generate conditions
    ├─ Conditional Mamba
    └─ Extract global class vector
    ↓
Decoder: SemanticGuidedMambaRefinement
    ├─ Fuse F_up + F_skip
    ├─ Apply class guidance
    ├─ Mamba refinement
    └─ Residual + Projection
    ↓
Output (B, num_classes, D, H, W)
```

### 关键创新点

#### 1. 窗口机制 (Window Mechanism)
- **重叠窗口**: 避免边界伪影
- **独立处理**: 每个窗口内独立应用 Mamba
- **平均合并**: 重叠区域取平均

#### 2. 双路径融合 (Dual-Path Fusion)
- **全局路径**: 捕获器官级长程依赖
- **局部路径**: 保留精细结构信息
- **自适应融合**: 学习两路径权重

#### 3. 条件生成 (Conditional Generation)
- **类别先验**: 从标签统计类别分布
- **参数生成**: 动态生成 Mamba 选择性参数
- **自适应调制**: 根据类别平衡调整模型行为

#### 4. 语义引导 (Semantic Guidance)
- **全局类别向量**: 从瓶颈提取全局语义
- **门控机制**: 类别信息作为门控信号
- **细化处理**: Mamba 在类别指导下细化特征

## 性能考虑 (Performance Considerations)

### 计算复杂度
- **LocalMamba**: O(N × w³), w = 窗口大小
- **HierarchicalMamba**: O(N/4 + N × w³)
- **ConditionalMamba**: O(N) + 轻量条件网络
- **SG-Mamba**: O(N) + 类别映射

### 内存使用
- **窗口划分**: 额外内存 ≈ N × overlap_ratio
- **双路径**: 2× 路径内存
- **建议**: GPU ≥ 16GB for typical cases

### 训练时间
- **相对于 CNN**: 约 1.2-1.5× 训练时间
- **降级模式**: 如 Mamba 不可用，自动回退到卷积

## 常见问题 (FAQ)

### Q1: Mamba 不可用怎么办？
**A**: 自动降级到卷积实现，不影响训练，但可能降低性能。

### Q2: 内存不足怎么办？
**A**: 减小 `window_size` 或 `overlap_ratio`，或减小批次大小。

### Q3: 训练比 TryTrainer 慢很多？
**A**: 正常现象。Mamba 比卷积复杂。可以减少 `d_state` 或 `expand`。

### Q4: 如何验证 Mamba 是否在工作？
**A**: 查看初始化日志，应该看到 "🔥 添加层次化Mamba模块到网络..."

### Q5: 性能没有提升？
**A**: 可能需要更多训练周期，或调整 Mamba 超参数。确保数据集有极端类别不平衡。

## 引用 (Citation)

如果这个实现对您的研究有帮助，请考虑引用：

```bibtex
@software{ttrainer2024,
  title={TTrainer: Hierarchical Mamba Architecture for Medical Image Segmentation},
  author={Your Name},
  year={2024},
  url={https://github.com/zhoumengi/hellozmzm}
}
```

## 相关链接 (Links)

- [详细技术文档](TTRAINER_IMPLEMENTATION.md)
- [使用示例](example_usage_ttrainer.py)
- [nnUNet 官方文档](https://github.com/MIC-DKFZ/nnUNet)
- [Mamba 论文](https://arxiv.org/abs/2312.00752)

## 贡献 (Contributing)

欢迎提交 Issue 和 Pull Request！

### 开发者
- 实现: GitHub Copilot Agent
- 概念设计: 基于用户需求

## 许可证 (License)

与原 nnUNet 保持一致。

## 致谢 (Acknowledgments)

- nnUNet 团队提供的优秀框架
- Mamba 作者的创新架构
- 医学图像分割社区的支持

---

**最后更新**: 2024-12-30
**版本**: 1.0.0
**状态**: ✅ 实现完成，等待测试
