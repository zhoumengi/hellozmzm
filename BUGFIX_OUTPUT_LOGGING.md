# Bug Fix: TTrainer Output Messages Not Appearing

## Issue

User reported that TTrainer initialization messages were not appearing in the terminal output, despite:
- Adding explicit `print()` statements with `flush=True`
- Writing to both stdout and stderr
- Using explicit `sys.stdout.flush()` and `sys.stderr.flush()` calls

The user correctly identified the root cause: **TTrainer.initialize_network() does call super().initialize_network(), but the output messages are not visible because nnUNet redirects stdout before the method is called.**

## Root Cause

The nnUNet framework redirects standard output streams (stdout) to its internal logging system before trainer initialization. This means:

1. Direct `print()` calls write to a redirected/captured stdout
2. Messages are captured but not displayed in the terminal
3. Only nnUNet's internal logger (`print_to_log_file`) produces visible terminal output

## Solution

Use nnUNet's official logging method `self.print_to_log_file()` for all output messages.

### Implementation

Created helper functions that output to multiple channels:

```python
def log_message(msg):
    # 1. Standard output (may be redirected)
    print(msg, flush=True)
    sys.stdout.flush()
    
    # 2. Standard error (usually not redirected)
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    
    # 3. KEY: Use nnUNet's official logger
    if hasattr(self, 'print_to_log_file'):
        self.print_to_log_file(msg)
```

### Changes Made

**File**: `LogWeightednnUNetTrainer.py`

1. **TTrainer.__init__()**: 
   - Created `log_init_message()` helper
   - All initialization messages now use this helper
   
2. **TTrainer.initialize_network()**:
   - Created `log_message()` helper  
   - All network initialization messages use this helper
   - Added explicit message: "✅ 父类网络初始化完成" after `super().initialize_network()`

3. **TTrainer._add_hierarchical_mamba_to_network()**:
   - Created `log_mamba_message()` helper
   - All Mamba module addition messages use this helper

## Why This Works

- `print_to_log_file()` is nnUNet's standard method for trainer output
- It writes to both the log file AND displays in terminal via nnUNet's logging system
- All official nnUNet trainers use this method for output
- Messages now appear exactly where users expect them during training

## Expected Output

Users will now see all TTrainer messages in their terminal:

```
🔥🔥🔥 初始化TTrainer - 层次化Mamba创新训练器 🔥🔥🔥
✅ 继承TryTrainer的hybrid注意力和损失函数
🎯 添加层次化Mamba创新:
   - 浅层(1-2): 局部感知Mamba
   - 中层(3): 层次化Mamba
   - 深层(瓶颈): 条件选择性Mamba
   - 解码器: 语义引导Mamba细化
✅✅✅ 检测到2D配置，将使用2D Mamba模块 ✅✅✅
📐 窗口大小: (8, 8)
⏳ Mamba模块将在网络初始化时添加...

🌟🌟🌟 开始网络初始化 - TTrainer 🌟🌟🌟
✅ 父类网络初始化完成
🚀🚀🚀 准备添加Mamba模块 (2D模式)... 🚀🚀🚀
🔥🔥🔥 添加层次化Mamba模块到网络 (2D模式)... 🔥🔥🔥

📍 阶段 1: 添加局部感知Mamba2D (channels=32)
  ✅ 局部感知Mamba2D已添加
📍 阶段 2: 添加局部感知Mamba2D (channels=64)
  ✅ 局部感知Mamba2D已添加
📍 阶段 3: 添加层次化Mamba2D (channels=128)
  ✅ 层次化Mamba2D已添加
📍 瓶颈层: 添加条件选择性Mamba2D (channels=512)
  ✅ 条件选择性Mamba2D已添加
📍 解码器: 添加语义引导Mamba2D细化模块
  ✅ 解码器阶段 0: SG-Mamba2D
  ✅ 解码器阶段 1: SG-Mamba2D
  [...etc...]

🎉🎉🎉 层次化Mamba2D模块添加完成! 🎉🎉🎉
  配置维度: 2D (H, W)
  窗口大小: (8, 8)
  总Mamba参数: xxx,xxx
  Mamba可用性: ✅ 使用mamba-ssm

✅✅✅ Mamba模块成功集成到网络中! ✅✅✅
```

## Commit

Commit: `a36b652` - "Use nnUNet logger (print_to_log_file) for all TTrainer output messages"

## Testing

To verify the fix works:
```bash
nnUNetv2_train DATASET_ID 2d FOLD -tr TTrainer
```

You should now see all initialization and network setup messages in the terminal output.
