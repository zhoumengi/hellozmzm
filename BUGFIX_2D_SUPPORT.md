# TTrainer 2D Configuration Support - Bug Fix

## Issue

When using TTrainer with a 2D configuration, the trainer failed to initialize properly because:

1. **Hardcoded channel assumptions**: The original implementation assumed 3D architecture with 5 stages `[32, 64, 128, 256, 320]`, but 2D configurations use 7 stages `[32, 64, 128, 256, 512, 512, 512]`.

2. **Missing 2D/3D detection**: No logic to detect whether the configuration is 2D or 3D, causing mismatches in network architecture.

3. **3D-only Mamba modules**: Window partition operations were designed for 3D tensors `(B, C, D, H, W)`, not compatible with 2D tensors `(B, C, H, W)`.

## Root Cause

From the user's configuration dump:
```
configuration_name: "2d"
n_stages: 7
features_per_stage: [32, 64, 128, 256, 512, 512, 512]
conv_op: torch.nn.modules.conv.Conv2d
```

The TTrainer tried to use:
- Hardcoded `encoder_channels = [32, 64, 128, 256, 320]` (wrong!)
- 3D window operations with `window_size = (4, 4, 4)` (incompatible with 2D!)

## Solution

### 1. Dynamic Channel Extraction

Added `_extract_network_channels()` method that:
- Extracts actual channel configuration from `configuration_manager`
- Falls back to intelligent defaults based on detected configuration type
- Handles both 2D and 3D architectures automatically

```python
def _extract_network_channels(self):
    """从实际网络架构中提取编码器和解码器通道数"""
    # Try to extract from configuration_manager
    if hasattr(self.configuration_manager, 'configuration'):
        features = config['architecture']['arch_kwargs']['features_per_stage']
        self.encoder_channels = features
        self.decoder_channels = list(reversed(features))
    
    # Fallback to defaults based on 2D vs 3D
    if '2d' in self.configuration_name.lower():
        self.encoder_channels = [32, 64, 128, 256, 512, 512, 512]  # 7 stages
    else:
        self.encoder_channels = [32, 64, 128, 256, 320]  # 5 stages
```

### 2. 2D/3D Configuration Detection

Modified `__init__` to detect configuration type and adjust settings:

```python
# Detect 2D vs 3D
is_2d = '2d' in self.configuration_name.lower()

if is_2d:
    # 2D: Disable Mamba, use CNN fallback
    self.mamba_config = {
        'use_mamba': False,
        'window_size': None,  # Not applicable for 2D
        ...
    }
else:
    # 3D: Use full Mamba
    self.mamba_config = {
        'use_mamba': True,
        'window_size': (4, 4, 4),
        ...
    }
```

### 3. Conditional Mamba Initialization

Updated `initialize_network()` to respect configuration:

```python
def initialize_network(self):
    super().initialize_network()
    self._extract_network_channels()
    
    # Only use Mamba if configuration allows and Mamba is available
    use_mamba = self.mamba_config.get('use_mamba', True) and MAMBA_AVAILABLE
    if use_mamba and not self.ttrainer_mamba_added:
        self._add_hierarchical_mamba_to_network()
    elif not use_mamba:
        print("⚠️ Mamba模块已禁用（2D配置），将使用标准卷积")
```

## Behavior Changes

### Before Fix:
- ❌ Crashes on 2D configurations
- ❌ Hardcoded channel assumptions
- ❌ No 2D/3D detection

### After Fix:
- ✅ Supports both 2D and 3D configurations
- ✅ Dynamically extracts channel information from actual network
- ✅ Automatically detects and adapts to configuration type
- ✅ For 2D: Uses CNN fallback (already implemented in Mamba modules)
- ✅ For 3D: Uses full hierarchical Mamba architecture

## Testing Recommendations

### Test Case 1: 2D Configuration
```bash
nnUNetv2_train DATASET 2d FOLD -tr TTrainer
```
Expected:
- Should initialize without errors
- Should print: "⚠️ Mamba模块已禁用（2D配置），将使用标准卷积"
- Should use CNN fallback in all Mamba modules

### Test Case 2: 3D Configuration
```bash
nnUNetv2_train DATASET 3d_fullres FOLD -tr TTrainer
```
Expected:
- Should initialize without errors
- Should print channel information from configuration
- Should add all Mamba modules (Local, Hierarchical, Conditional, SG-Mamba)

## Backward Compatibility

- ✅ Existing 3D configurations continue to work as before
- ✅ No changes to Mamba module APIs
- ✅ Fallback mechanisms unchanged
- ✅ All documentation remains valid

## Future Improvements

Consider implementing 2D-specific Mamba modules:
- `LocalMamba2D`: Window partition for (H, W) instead of (D, H, W)
- `HierarchicalMamba2D`: Adapted for 2D feature maps
- This would enable Mamba benefits for 2D datasets

However, current fallback to CNN is safe and functional.
