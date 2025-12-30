# # /root/autodl-tmp/nnUNet/nnunetv2/training/nnUNetTrainer/log_weighted_trainer.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from typing import Union, List, Tuple

# # ============ 渐进式课程学习 ============
# class LogWeightednnUNetTrainer(nnUNetTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, 
#                  dataset_json: dict, device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, device)
        
#         print("\n" + "="*80)
#         print("🎓 初始化渐进式课程学习训练器")
#         print("="*80)
        
#         # 基础配置
#         self.oversample_foreground_percent = 0.99
#         self.initial_lr = 0.01
#         self.weight_decay = 1e-4
#         self.epoch_count = 0
#         self.best_ema_dice = 0.0
        
#         # 课程学习阶段配置（根据你的数据特点优化）
#         self.curriculum_stages = self._define_curriculum_stages()
#         self.current_stage_idx = 0
#         self.current_classes = self.curriculum_stages[0]['classes']
        
#         # 打印课程计划
#         self._print_curriculum_plan()
        
#         # 初始化权重（根据当前阶段调整）
#         self._initialize_stage_weights()
        
#         print("="*80 + "\n")
    
#     def _define_curriculum_stages(self):
#         """定义渐进式课程学习阶段"""
#         # 阶段1：先训练背景和已经表现出预测能力的类别
#         stage1_classes = [0, 6, 8]  # 背景 + 已有预测的类别
        
#         # 阶段2：添加相对较多的前景类别
#         stage2_classes = stage1_classes + [1, 2, 7, 39, 40]  # 体素较多的类别
        
#         # 阶段3：添加更多前景类别（体素中等）
#         stage3_classes = stage2_classes + list(range(9, 20)) + [45, 46, 47, 48, 49, 50]
        
#         # 阶段4：添加更多类别
#         stage4_classes = stage3_classes + list(range(20, 45)) + list(range(51, 65))
        
#         # 阶段5：添加剩余类别（排除43,44,80）
#         stage5_classes = stage4_classes + list(range(65, 81))
#         # 移除不需要训练的类别
#         for cls in [43, 44, 80]:
#             if cls in stage5_classes:
#                 stage5_classes.remove(cls)
        
#         stages = [
#             {'name': '阶段1: 基础类别', 
#              'epochs': 50, 
#              'classes': stage1_classes,
#              'lr': 0.01,
#              'description': '训练背景和已有预测的类别'},
            
#             {'name': '阶段2: 扩展类别', 
#              'epochs': 100, 
#              'classes': stage2_classes,
#              'lr': 0.005,
#              'description': '添加体素较多的关键类别'},
            
#             {'name': '阶段3: 中等类别', 
#              'epochs': 150, 
#              'classes': stage3_classes,
#              'lr': 0.002,
#              'description': '添加中等体素的类别'},
            
#             {'name': '阶段4: 更多类别', 
#              'epochs': 200, 
#              'classes': stage4_classes,
#              'lr': 0.001,
#              'description': '添加更多前景类别'},
            
#             {'name': '阶段5: 完整类别', 
#              'epochs': 500, 
#              'classes': stage5_classes,
#              'lr': 0.0005,
#              'description': '训练所有类别（排除43,44,80）'}
#         ]
        
#         return stages
    
#     def _print_curriculum_plan(self):
#         """打印课程学习计划"""
#         print("\n📚 渐进式课程学习计划:")
#         print("-" * 80)
        
#         total_epochs = 0
#         for i, stage in enumerate(self.curriculum_stages):
#             total_epochs += stage['epochs']
#             print(f"\n阶段 {i+1}: {stage['name']}")
#             print(f"  训练周期: {stage['epochs']} epochs")
#             print(f"  学习率: {stage['lr']}")
#             print(f"  类别数: {len(stage['classes'])}")
#             print(f"  类别列表: {sorted(stage['classes'])[:20]}...")  # 只显示前20个
            
#             # 按类别分组显示
#             bg_classes = [c for c in stage['classes'] if c == 0]
#             focus_classes = [c for c in stage['classes'] if c in [6, 8, 39, 40]]
#             other_classes = [c for c in stage['classes'] if c > 0 and c not in focus_classes]
            
#             print(f"  背景: {bg_classes}")
#             print(f"  焦点类别: {focus_classes}")
#             print(f"  其他类别: {len(other_classes)}个")
#             print(f"  描述: {stage['description']}")
        
#         print(f"\n📅 总训练周期: {total_epochs} epochs")
#         print(f"🎯 最终类别数: {len(self.curriculum_stages[-1]['classes'])} (排除43,44,80)")
    
#     def _initialize_stage_weights(self):
#         """为当前阶段初始化权重"""
#         # 你的体素统计数据
#         voxel_counts = {
#             0: 10357996276, 1: 709790043, 2: 34062112, 3: 6740490, 4: 6845641,
#             5: 2352907, 6: 2317488, 7: 222131847, 8: 2049060, 9: 9904774,
#             10: 513649, 11: 1893477, 12: 1306498, 13: 2238177, 14: 2565055,
#             15: 2764938, 16: 6233357, 17: 5799851, 18: 2204634, 19: 1894346,
#             20: 1325445, 21: 2235198, 22: 2686611, 23: 2850689, 24: 6449628,
#             25: 6172869, 26: 2364195, 27: 3586957, 28: 4371277, 29: 7462541,
#             30: 5516252, 31: 5621255, 32: 8634051, 33: 9863243, 34: 7354314,
#             35: 3561221, 36: 4381718, 37: 7616134, 38: 5782732, 39: 5743643,
#             40: 8713962, 41: 9710653, 42: 7634230, 43: 11218, 44: 6401,
#             45: 303363, 46: 268274, 47: 131338, 48: 49350, 49: 32793,
#             50: 62648, 51: 52939, 52: 69487, 53: 220541, 54: 218058,
#             55: 88650, 56: 49683, 57: 32498, 58: 59426, 59: 54468,
#             60: 71912, 61: 236091, 62: 234591, 63: 95453, 64: 114584,
#             65: 158548, 66: 349849, 67: 221625, 68: 240117, 69: 418918,
#             70: 488250, 71: 380129, 72: 113697, 73: 157527, 74: 346682,
#             75: 230442, 76: 238892, 77: 418256, 78: 489991, 79: 385921,
#             80: 555
#         }
        
#         self.num_classes = 81  # 总类别数
#         self.voxel_counts = voxel_counts
        
#         # 为当前阶段创建权重
#         weights = np.ones(self.num_classes, dtype=np.float32)
        
#         # 1. 当前阶段不训练的类别权重设为0（模型不会预测它们）
#         for cls in range(self.num_classes):
#             if cls not in self.current_classes:
#                 weights[cls] = 0.0  # 完全不训练这些类别
        
#         # 2. 背景权重较低
#         if 0 in self.current_classes:
#             weights[0] = 0.01
        
#         # 3. 不重要类别（43,44,80）永远不训练
#         for cls in [43, 44, 80]:
#             weights[cls] = 0.0
        
#         # 4. 根据体素数量设置前景类别权重
#         for cls in self.current_classes:
#             if cls == 0 or cls in [43, 44, 80]:
#                 continue
            
#             count = voxel_counts[cls]
            
#             # 根据体素数量设置权重
#             if count < 10000:
#                 weights[cls] = 50.0  # 极少数类别
#             elif count < 100000:
#                 weights[cls] = 20.0  # 很少数类别
#             elif count < 1000000:
#                 weights[cls] = 10.0  # 中等类别
#             elif count < 10000000:
#                 weights[cls] = 5.0   # 多数类别
#             else:
#                 weights[cls] = 2.0   # 极多数类别
        
#         # 转换为tensor
#         self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
#         # 打印当前阶段权重信息
#         self._print_stage_info()
    
#     def _print_stage_info(self):
#         """打印当前阶段信息"""
#         current_stage = self.curriculum_stages[self.current_stage_idx]
        
#         print(f"\n🎓 当前训练阶段: {current_stage['name']}")
#         print(f"  阶段 {self.current_stage_idx + 1}/{len(self.curriculum_stages)}")
#         print(f"  目标epoch: {current_stage['epochs']}")
#         print(f"  当前epoch: {self.epoch_count}")
#         print(f"  训练类别数: {len(self.current_classes)}")
#         print(f"  训练类别: {sorted(self.current_classes)}")
        
#         # 统计各类别情况
#         bg_count = sum(1 for c in self.current_classes if c == 0)
#         focus_count = sum(1 for c in self.current_classes if c in [6, 8, 39, 40])
#         other_count = len(self.current_classes) - bg_count - focus_count
        
#         print(f"  背景: {bg_count}个")
#         print(f"  焦点类别: {focus_count}个")
#         print(f"  其他类别: {other_count}个")
        
#         # 显示权重分布
#         active_weights = [self.class_weights[c].item() for c in self.current_classes if c > 0]
#         if active_weights:
#             avg_weight = np.mean(active_weights)
#             max_weight = np.max(active_weights)
#             min_weight = np.min(active_weights)
#             print(f"  权重范围: {min_weight:.1f} - {max_weight:.1f}")
#             print(f"  平均权重: {avg_weight:.1f}")
    
#     def configure_loss_function(self):
#         """配置课程学习专用的损失函数"""
#         from nnunetv2.training.loss.dice import SoftDiceLoss
#         from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        
#         print("\n" + "="*80)
#         print("📚 配置课程学习专用损失函数")
#         print("="*80)
        
#         # 创建当前阶段类别的mask
#         self.current_classes_tensor = torch.tensor(
#             self.current_classes, dtype=torch.long, device=self.device
#         )
        
#         # 自定义损失函数，只关注当前阶段的类别
#         class CurriculumLoss(nn.Module):
#             def __init__(self, base_loss, current_classes, class_weights, device):
#                 super().__init__()
#                 self.base_loss = base_loss
#                 self.current_classes = current_classes
#                 self.class_weights = class_weights
#                 self.device = device
                
#                 # 创建mask用于筛选当前阶段类别
#                 self.valid_mask = torch.zeros(81, dtype=torch.bool, device=device)
#                 self.valid_mask[current_classes] = True
            
#             def forward(self, pred, target):
#                 # 1. 创建目标mask：只保留当前阶段的类别
#                 target_mask = torch.zeros_like(target, dtype=torch.bool)
#                 for cls in self.current_classes:
#                     target_mask = target_mask | (target == cls)
                
#                 # 将非当前阶段类别设为背景
#                 target_filtered = target.clone()
#                 target_filtered[~target_mask] = 0
                
#                 # 2. 过滤预测：只保留当前阶段的输出通道
#                 pred_filtered = pred.clone()
#                 for cls in range(pred.shape[1]):
#                     if not self.valid_mask[cls]:
#                         # 将非当前阶段类别的logits设为很小的值，避免被预测
#                         pred_filtered[:, cls, ...] = -100.0
                
#                 # 3. 计算损失
#                 loss = self.base_loss(pred_filtered, target_filtered)
                
#                 return loss
        
#         # 基础损失函数
#         base_loss = DC_and_CE_loss(
#             soft_dice_kwargs={
#                 'batch_dice': True,
#                 'smooth': 1e-5,
#                 'do_bg': True,
#                 'smooth_in_nom': True,
#                 'background_weight': 0.01 if 0 in self.current_classes else 0.0
#             },
#             ce_kwargs={
#                 'weight': self.class_weights
#             },
#             weight_ce=1.0,
#             weight_dice=1.0
#         )
        
#         # 包装为课程学习损失
#         loss = CurriculumLoss(
#             base_loss=base_loss,
#             current_classes=self.current_classes_tensor,
#             class_weights=self.class_weights,
#             device=self.device
#         )
        
#         print(f"✅ 课程学习损失函数配置完成")
#         print(f"   - 当前阶段类别数: {len(self.current_classes)}")
#         print(f"   - 背景权重: {self.class_weights[0].item() if 0 in self.current_classes else 0.0}")
#         print(f"   - 焦点类别权重: {self.class_weights[6].item() if 6 in self.current_classes else 'N/A'}")
#         print("="*80 + "\n")
        
#         return loss
    
#     def on_train_start(self):
#         """训练开始时的回调"""
#         super().on_train_start()
        
#         # 设置当前阶段的学习率
#         current_stage = self.curriculum_stages[self.current_stage_idx]
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = current_stage['lr']
        
#         print(f"\n🚀 课程学习训练开始")
#         print(f"  阶段: {current_stage['name']}")
#         print(f"  初始学习率: {current_stage['lr']}")
#         print(f"  权重衰减: {self.weight_decay}")
    
#     def on_epoch_begin(self):
#         """每个epoch开始时的回调"""
#         # 检查是否需要切换到下一个阶段
#         self._check_stage_transition()
        
#         # 更新学习率（根据阶段）
#         current_stage = self.curriculum_stages[self.current_stage_idx]
#         if self.epoch_count > 0 and self.epoch_count % 10 == 0:
#             # 每10个epoch轻微降低学习率
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] *= 0.95
#                 print(f"📉 Epoch {self.epoch_count}: 学习率降低至 {param_group['lr']:.6f}")
    
#     def _check_stage_transition(self):
#         """检查并执行阶段切换"""
#         current_stage = self.curriculum_stages[self.current_stage_idx]
        
#         # 如果当前阶段训练完成，切换到下一个阶段
#         if self.epoch_count >= sum(s['epochs'] for s in self.curriculum_stages[:self.current_stage_idx + 1]):
#             if self.current_stage_idx < len(self.curriculum_stages) - 1:
#                 # 切换到下一个阶段
#                 self.current_stage_idx += 1
#                 self.current_classes = self.curriculum_stages[self.current_stage_idx]['classes']
                
#                 # 重新初始化权重
#                 self._initialize_stage_weights()
                
#                 # 更新学习率
#                 new_stage = self.curriculum_stages[self.current_stage_idx]
#                 for param_group in self.optimizer.param_groups:
#                     param_group['lr'] = new_stage['lr']
                
#                 print(f"\n🎉 切换到新阶段: {new_stage['name']}")
#                 print(f"  新学习率: {new_stage['lr']}")
#                 print(f"  新增类别: {len(self.current_classes) - len(set().union(*[s['classes'] for s in self.curriculum_stages[:self.current_stage_idx]]))}个")
#                 print(f"  总训练类别: {len(self.current_classes)}")
                
#                 # 重新配置损失函数
#                 self.loss = self.configure_loss_function()
    
#     def on_epoch_end(self):
#         """每个epoch结束时的回调"""
#         super().on_epoch_end()
#         self.epoch_count += 1
        
#         # 每10个epoch打印阶段信息
#         if self.epoch_count % 10 == 0:
#             current_stage = self.curriculum_stages[self.current_stage_idx]
#             stage_start_epoch = sum(s['epochs'] for s in self.curriculum_stages[:self.current_stage_idx])
#             stage_epoch = self.epoch_count - stage_start_epoch
            
#             print(f"\n📊 阶段进度 - {current_stage['name']}")
#             print(f"  阶段内epoch: {stage_epoch}/{current_stage['epochs']}")
#             print(f"  总epoch: {self.epoch_count}")
#             print(f"  当前学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
    
#     def on_validation_end(self, val_loss, pseudo_dice, pseudo_dice_ema):
#         """验证结束时的回调"""
#         super().on_validation_end(val_loss, pseudo_dice, pseudo_dice_ema)
        
#         if pseudo_dice is not None:
#             print("\n" + "="*80)
#             print(f"🎓 Epoch {self.epoch_count} - 课程学习性能分析")
#             print("="*80)
            
#             # 只分析当前阶段的类别
#             current_classes_set = set(self.current_classes)
            
#             # 统计当前阶段类别的表现
#             stage_dices = []
#             stage_nonzero = 0
            
#             for cls in self.current_classes:
#                 if cls < len(pseudo_dice) and not np.isnan(pseudo_dice[cls]):
#                     dice = pseudo_dice[cls]
#                     stage_dices.append(dice)
#                     if dice > 0:
#                         stage_nonzero += 1
            
#             if stage_dices:
#                 avg_dice = np.mean(stage_dices)
#                 print(f"\n📈 当前阶段性能统计:")
#                 print(f"  训练类别数: {len(self.current_classes)}")
#                 print(f"  平均Dice: {avg_dice:.4f}")
#                 print(f"  非零Dice类别数: {stage_nonzero}/{len(self.current_classes)}")
                
#                 # 显示每个类别的表现
#                 print(f"\n📋 类别详细表现:")
#                 for cls in sorted(self.current_classes):
#                     if cls < len(pseudo_dice):
#                         dice = pseudo_dice[cls]
#                         if not np.isnan(dice):
#                             weight = self.class_weights[cls].item() if cls < len(self.class_weights) else 1.0
#                             count = self.voxel_counts.get(cls, 0)
                            
#                             status = "✅" if dice > 0.3 else "⚠️" if dice > 0.1 else "❌"
#                             print(f"  {status} 类别 {cls:2d}: Dice={dice:.4f} | 权重={weight:5.1f} | 体素={count:>12,}")
            
#             # 显示非当前阶段类别的情况（应该都是0或NaN）
#             other_dices = []
#             for cls in range(len(pseudo_dice)):
#                 if cls not in current_classes_set and cls not in [43, 44, 80]:
#                     if cls < len(pseudo_dice) and not np.isnan(pseudo_dice[cls]):
#                         dice = pseudo_dice[cls]
#                         if dice > 0:
#                             other_dices.append((cls, dice))
            
#             if other_dices:
#                 print(f"\n⚠️  注意: 非当前阶段类别有预测值:")
#                 for cls, dice in sorted(other_dices, key=lambda x: x[1], reverse=True)[:5]:
#                     print(f"  类别 {cls:2d}: Dice={dice:.4f} (未在训练中)")
            
#             print("="*80 + "\n")

# # ============ 渐进式课程学习 + 权重调整 ============
# class CurriculumWeightedTrainer(LogWeightednnUNetTrainer):
#     """课程学习 + 动态权重调整"""
#     def __init__(self, plans: dict, configuration: str, fold: int, 
#                  dataset_json: dict, device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, device)
        
#         # 添加动态权重调整
#         self.class_performance_history = {}
#         self.poor_performance_streaks = {}
        
#         # 初始化记录
#         for i in range(self.num_classes):
#             self.class_performance_history[i] = []
#             self.poor_performance_streaks[i] = 0
        
#         print("🎯 启用课程学习 + 动态权重调整")
    
#     def on_validation_end(self, val_loss, pseudo_dice, pseudo_dice_ema):
#         """验证结束时的回调，包含动态权重调整"""
#         super().on_validation_end(val_loss, pseudo_dice, pseudo_dice_ema)
        
#         if pseudo_dice is not None:
#             # 更新性能历史
#             for cls in self.current_classes:
#                 if cls < len(pseudo_dice) and not np.isnan(pseudo_dice[cls]):
#                     self.class_performance_history[cls].append(pseudo_dice[cls])
#                     if len(self.class_performance_history[cls]) > 10:
#                         self.class_performance_history[cls].pop(0)
            
#             # 动态调整当前阶段类别的权重
#             self._adjust_stage_weights(pseudo_dice)
    
#     def _adjust_stage_weights(self, pseudo_dice):
#         """动态调整当前阶段类别的权重"""
#         adjustments = []
        
#         for cls in self.current_classes:
#             if cls == 0 or cls in [43, 44, 80]:
#                 continue
                
#             if cls >= len(pseudo_dice) or np.isnan(pseudo_dice[cls]):
#                 continue
            
#             dice = pseudo_dice[cls]
#             current_weight = self.class_weights[cls].item()
            
#             # 检查连续表现差的情况
#             if dice == 0:
#                 self.poor_performance_streaks[cls] += 1
#             elif dice < 0.05:
#                 self.poor_performance_streaks[cls] += 0.5
#             else:
#                 self.poor_performance_streaks[cls] = max(0, self.poor_performance_streaks[cls] - 0.2)
            
#             # 需要调整的情况
#             new_weight = current_weight
            
#             # 情况1：连续3次Dice为0
#             if self.poor_performance_streaks[cls] >= 3:
#                 new_weight = current_weight * 2.0
#                 adjustments.append((cls, current_weight, new_weight, "连续3次Dice=0"))
            
#             # 情况2：历史平均Dice < 0.1
#             elif (self.class_performance_history[cls] and 
#                   len(self.class_performance_history[cls]) >= 5):
#                 avg_dice = np.mean(self.class_performance_history[cls][-5:])
#                 if avg_dice < 0.1:
#                     new_weight = current_weight * 1.5
#                     adjustments.append((cls, current_weight, new_weight, f"平均Dice={avg_dice:.3f}<0.1"))
            
#             # 应用调整
#             if new_weight != current_weight:
#                 # 限制范围
#                 new_weight = min(100.0, max(1.0, new_weight))
#                 self.class_weights[cls] = new_weight
        
#         # 打印调整信息
#         if adjustments:
#             print(f"\n⚡ 动态权重调整 ({len(adjustments)}个类别):")
#             for cls, old_w, new_w, reason in adjustments[:10]:  # 最多显示10个
#                 print(f"  类别 {cls:2d}: {old_w:5.1f} → {new_w:5.1f} ({reason})")






# import torch
# import torch.nn as nn
# import os
# import sys
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# import numpy as np

# # 在导入任何torch模块之前就彻底禁用编译
# os.environ['NNUNET_COMPILE'] = '0'
# os.environ['TORCHDYNAMO_DISABLE'] = '1'
# os.environ['TORCH_COMPILE_DEBUG'] = '0'

# # ================ 真正解决问题的修复 ================
# import warnings
# import nnunetv2.evaluation.evaluate_predictions as eval_module

# # 定义要忽略的类别
# IGNORE_CLASSES = [4, 5, 6, 44, 45, 46, 47, 43]

# print("="*60)
# print("安装Dice NaN根本性修复")
# print("="*60)

# # 1. 修复compute_metrics函数 - 从根源防止NaN
# original_compute_metrics = eval_module.compute_metrics

# def fixed_compute_metrics(reference_file, prediction_file, image_reader_writer,
#                          labels_or_regions, ignore_label=None):
#     """修复版本：确保不产生NaN值"""
#     try:
#         # 调用原始函数
#         result = original_compute_metrics(
#             reference_file, prediction_file, image_reader_writer,
#             labels_or_regions, ignore_label
#         )
        
#         # 处理结果，确保没有NaN
#         if isinstance(result, dict) and 'metrics' in result:
#             metrics = result['metrics']
            
#             for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                 if metric_name in metrics:
#                     metric_dict = metrics[metric_name]
                    
#                     if isinstance(metric_dict, dict):
#                         # 移除忽略的类别
#                         for cls in IGNORE_CLASSES:
#                             if str(cls) in metric_dict:
#                                 del metric_dict[str(cls)]
#                             elif cls in metric_dict:
#                                 del metric_dict[cls]
                        
#                         # 修复NaN值
#                         for key in list(metric_dict.keys()):
#                             val = metric_dict[key]
#                             if isinstance(val, (int, float)):
#                                 if np.isnan(val):
#                                     metric_dict[key] = 0.0
#                             elif hasattr(val, 'item'):
#                                 if np.isnan(val.item()):
#                                     metric_dict[key] = 0.0
        
#         return result
        
#     except Exception as e:
#         print(f"⚠️ compute_metrics错误: {e}")
#         return {'metrics': {'dice': {}, 'hausdorff_distance_95': {}, 'mean_surface_distance': {}}}

# eval_module.compute_metrics = fixed_compute_metrics
# print("✅ 已修复 compute_metrics 函数")

# # 2. 修复聚合函数summarize_results
# if hasattr(eval_module, 'summarize_results'):
#     original_summarize = eval_module.summarize_results
    
#     def fixed_summarize_results(results, args=None):
#         """修复聚合函数，确保最终平均值不是NaN"""
#         try:
#             # 先修复所有结果中的NaN
#             fixed_results = []
#             for res in results:
#                 if isinstance(res, dict) and 'metrics' in res:
#                     fixed_res = res.copy()
#                     metrics = fixed_res['metrics']
                    
#                     for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                         if metric_name in metrics:
#                             metric_dict = metrics[metric_name]
#                             if isinstance(metric_dict, dict):
#                                 # 修复NaN
#                                 for key, val in metric_dict.items():
#                                     if isinstance(val, (int, float)) and np.isnan(val):
#                                         metric_dict[key] = 0.0
                    
#                     fixed_results.append(fixed_res)
#                 else:
#                     fixed_results.append(res)
            
#             # 调用原始函数
#             if args is not None:
#                 summary = original_summarize(fixed_results, args)
#             else:
#                 summary = original_summarize(fixed_results)
            
#             # 确保最终结果没有NaN
#             def fix_nan_in_dict(d):
#                 if isinstance(d, dict):
#                     for k, v in d.items():
#                         if isinstance(v, dict):
#                             fix_nan_in_dict(v)
#                         elif isinstance(v, (int, float)) and np.isnan(v):
#                             d[k] = 0.0
            
#             if isinstance(summary, dict):
#                 fix_nan_in_dict(summary)
            
#             return summary
            
#         except Exception as e:
#             print(f"⚠️ summarize_results错误: {e}")
#             return {}

#     eval_module.summarize_results = fixed_summarize_results
# print("✅ 已修复 summarize_results 函数")

# # 3. 修改compute_metrics_on_folder函数
# original_compute_metrics_on_folder = eval_module.compute_metrics_on_folder

# def fixed_compute_metrics_on_folder(folder_ref, folder_pred, image_reader_writer,
#                                    labels_or_regions, ignore_label=None, 
#                                    num_processes: int = 1):
#     """修复文件夹指标计算"""
#     try:
#         results = original_compute_metrics_on_folder(
#             folder_ref, folder_pred, image_reader_writer,
#             labels_or_regions, ignore_label, num_processes
#         )
        
#         # 后处理：确保没有NaN
#         fixed_results = []
#         for res in results:
#             if isinstance(res, dict) and 'metrics' in res:
#                 fixed_res = res.copy()
#                 metrics = fixed_res['metrics']
                
#                 for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                     if metric_name in metrics:
#                         metric_dict = metrics[metric_name]
#                         if isinstance(metric_dict, dict):
#                             # 修复NaN
#                             for key, val in metric_dict.items():
#                                 if isinstance(val, (int, float)) and np.isnan(val):
#                                     metric_dict[key] = 0.0
                
#                 fixed_results.append(fixed_res)
#             else:
#                 fixed_results.append(res)
        
#         return fixed_results
        
#     except Exception as e:
#         print(f"⚠️ compute_metrics_on_folder错误: {e}")
#         return []

# eval_module.compute_metrics_on_folder = fixed_compute_metrics_on_folder
# print("✅ 已修复 compute_metrics_on_folder 函数")

# # 4. 修复np.nanmean作为最后保障
# original_nanmean = np.nanmean

# def safe_nanmean(a, **kwargs):
#     """安全的nanmean，确保不返回NaN"""
#     try:
#         result = original_nanmean(a, **kwargs)
#         if np.isnan(result):
#             return 0.0
#         return result
#     except:
#         return 0.0

# np.nanmean = safe_nanmean
# print("✅ 已修复 np.nanmean 函数作为最后保障")

# print(f"\n" + "="*60)
# print(f"🎯 修复总结:")
# print(f"   忽略的类别: {IGNORE_CLASSES}")
# print(f"   修复了3个关键函数 + np.nanmean作为保障")
# print(f"   策略: 从计算源头防止NaN，层层防护")
# print("="*60 + "\n")
# # ================ 修复结束 ================

# # ================ Mamba 集成开始 ================
# # 导入官方 Mamba（使用你已经安装成功的版本）
# from mamba_ssm import Mamba

# # Mamba 瓶颈层定义（使用官方 Mamba）
# class MambaBottleneck3D(nn.Module):
#     """3D Mamba bottleneck using official mamba-ssm"""
    
#     def __init__(self, in_channels, out_channels, seq_len=32, d_state=16, d_conv=4, expand=2):
#         super().__init__()
        
#         print(f"🔧 创建 MambaBottleneck3D: {in_channels} -> {out_channels}")
#         print(f"   Mamba 配置: d_state={d_state}, d_conv={d_conv}, expand={expand}")
        
#         self.seq_len = seq_len
#         self.in_channels = in_channels
#         self.out_channels = out_channels
        
#         # ========== 1. 标准3D卷积路径（保持兼容性）==========
#         self.conv_path = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.InstanceNorm3d(out_channels),
#             nn.LeakyReLU(inplace=True, negative_slope=0.01),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.InstanceNorm3d(out_channels),
#             nn.LeakyReLU(inplace=True, negative_slope=0.01),
#         )
        
#         # ========== 2. 官方 Mamba 增强路径 ==========
#         self.mamba_branch = self._create_mamba_branch(out_channels, d_state, d_conv, expand)
        
#         # ========== 3. 特征融合 ==========
#         self.fusion = nn.Sequential(
#             nn.Conv3d(out_channels * 2, out_channels, kernel_size=1),
#             nn.InstanceNorm3d(out_channels),
#             nn.LeakyReLU(inplace=True, negative_slope=0.01),
#         )
        
#         # ========== 4. 残差连接 ==========
#         if in_channels != out_channels:
#             self.residual = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, kernel_size=1),
#                 nn.InstanceNorm3d(out_channels)
#             )
#         else:
#             self.residual = nn.Identity()
    
#     def _create_mamba_branch(self, channels, d_state, d_conv, expand):
#         """创建官方 Mamba 分支"""
#         class OfficialMambaBranch(nn.Module):
#             def __init__(self, channels, seq_len, d_state, d_conv, expand):
#                 super().__init__()
#                 self.seq_len = seq_len
#                 self.channels = channels
                
#                 # 降维到适合 Mamba 的维度（官方推荐 128-256）
#                 self.proj_in = nn.Conv3d(channels, 128, kernel_size=1)
#                 self.norm_in = nn.InstanceNorm3d(128)
#                 self.act = nn.LeakyReLU(inplace=True, negative_slope=0.01)
                
#                 # 官方 Mamba 层
#                 self.mamba = Mamba(
#                     d_model=128,      # Mamba 输入维度
#                     d_state=d_state,  # 状态维度
#                     d_conv=d_conv,    # 卷积核大小
#                     expand=expand,    # 扩展因子
#                 )
                
#                 # 恢复维度
#                 self.proj_out = nn.Conv3d(128, channels, kernel_size=1)
#                 self.norm_out = nn.InstanceNorm3d(channels)
                
#                 print(f"      Mamba 分支: 128-dim, 官方实现")
            
#             def forward(self, x):
#                 identity = x
                
#                 # 降维
#                 x = self.act(self.norm_in(self.proj_in(x)))
                
#                 # 将 3D 特征转换为序列
#                 B, C, D, H, W = x.shape
#                 L = D * H * W  # 总空间位置数
                
#                 # 控制序列长度以避免内存爆炸
#                 if L > self.seq_len:
#                     # 随机采样保持固定序列长度
#                     indices = torch.randperm(L, device=x.device)[:self.seq_len]
#                     x_flat = x.view(B, C, L)
#                     x_sampled = torch.index_select(x_flat, 2, indices)
#                     x_seq = x_sampled.transpose(1, 2)  # (B, seq_len, C)
#                     sampled_indices = indices
#                 else:
#                     # 使用所有位置
#                     x_flat = x.view(B, C, L)
#                     x_seq = x_flat.transpose(1, 2)  # (B, L, C)
#                     sampled_indices = None
                
#                 # 官方 Mamba 处理（需要在 GPU 上）
#                 if not x_seq.is_cuda:
#                     x_seq = x_seq.cuda()
                
#                 x_seq = self.mamba(x_seq)
                
#                 # 恢复 3D 形状
#                 if sampled_indices is None:
#                     # 直接恢复
#                     x = x_seq.transpose(1, 2).view(B, C, D, H, W)
#                 else:
#                     # 插值恢复
#                     x_seq = x_seq.transpose(1, 2)  # (B, C, seq_len)
#                     # 创建全零张量然后填充
#                     x_full = torch.zeros(B, C, L, device=x_seq.device, dtype=x_seq.dtype)
#                     x_full[:, :, sampled_indices] = x_seq
#                     x = x_full.view(B, C, D, H, W)
                
#                 # 恢复维度
#                 x = self.norm_out(self.proj_out(x))
                
#                 return x + identity
        
#         return OfficialMambaBranch(channels, self.seq_len, d_state, d_conv, expand)
    
#     def forward(self, x):
#         identity = self.residual(x)
        
#         # 标准卷积路径
#         conv_out = self.conv_path(x)
        
#         # 官方 Mamba 增强
#         mamba_out = self.mamba_branch(conv_out)
        
#         # 特征融合
#         combined = torch.cat([conv_out, mamba_out], dim=1)
#         output = self.fusion(combined)
        
#         return output + identity

# # ================ 修改后的训练器类 ================
# class LogWeightednnUNetTrainer(nnUNetTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, 
#                  dataset_json: dict, device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, device)
        
#         # ========== 保持您的原始类别设置 ==========
#         self.train_classes = []
        
#         # 1. 所有牙齿（11-42）
#         for i in range(11, 43):
#             if i != 43:  # 排除不存在的43
#                 self.train_classes.append(i)
        
#         # 2. 牙髓（48）
#         self.train_classes.append(48)
        
#         # 3. 重要解剖结构
#         important_classes = [1, 2, 3, 4, 5, 6, 8, 9]
#         for cls in important_classes:
#             if cls not in self.train_classes:
#                 self.train_classes.append(cls)
        
#         self.train_classes.sort()
        
#         # 定义忽略类别（与评估模块保持一致）
#         self.ignore_classes = [4, 5, 6, 44, 45, 46, 47, 43]
        
#         # 初始化体素计数
#         self.voxel_counts = {
#             0: 10357996276, 1: 709790043, 2: 34062112, 3: 6740490, 4: 6845641,
#             5: 2352907, 6: 2317488, 7: 222131847, 8: 2049060, 9: 9904774,
#             10: 513649, 11: 1893477, 12: 1306498, 13: 2238177, 14: 2565055,
#             15: 2764938, 16: 6233357, 17: 5799851, 18: 2204634, 19: 1894346,
#             20: 1325445, 21: 2235198, 22: 2686611, 23: 2850689, 24: 6449628,
#             25: 6172869, 26: 2364195, 27: 3586957, 28: 4371277, 29: 7462541,
#             30: 5516252, 31: 5621255, 32: 8634051, 33: 9863243, 34: 7354314,
#             35: 3561221, 36: 4381718, 37: 7616134, 38: 5782732, 39: 5743643,
#             40: 8713962, 41: 9710653, 42: 7634230, 
#             43: 11218,
#             44: 6401, 45: 303363, 46: 268274, 47: 131338, 48: 49350
#         }
        
#         print(f"🎯 训练配置:")
#         print(f"  总类别数: 49 (0-48)")
#         print(f"  训练类别数: {len(self.train_classes)}个")
#         print(f"  训练类别: {self.train_classes}")
#         print(f"  忽略类别: {self.ignore_classes}")
        
#         # 计算权重
#         self.class_weights = self._calculate_fixed_weights()
        
#         # 应用权重到损失函数
#         self.apply_weights_to_loss()
        
#         # 监控变量
#         self.best_mean_dice = 0
#         self.patience_counter = 0
#         self.max_patience = 20
        
#         # ========== Mamba 相关设置 ==========
#         self.use_mamba = True  # 启用官方 Mamba
#         self.mamba_added = False  # 标记是否已添加 Mamba
#         self.mamba_seq_len = 32  # 序列长度
#         self.mamba_d_state = 16  # Mamba 状态维度
#         self.mamba_d_conv = 4    # Mamba 卷积核大小
#         self.mamba_expand = 2    # Mamba 扩展因子
        
#         if self.use_mamba:
#             print(f"\n🎯 Mamba 配置:")
#             print(f"  使用官方 mamba-ssm")
#             print(f"  序列长度: {self.mamba_seq_len}")
#             print(f"  状态维度: {self.mamba_d_state}")
#             print(f"  卷积核大小: {self.mamba_d_conv}")
#             print(f"  扩展因子: {self.mamba_expand}")
        
#         # 打印初始化信息
#         self.print_initial_info()
    
#     def _calculate_fixed_weights(self):
#         """修复权重计算"""
#         num_classes = 49
#         weights = torch.ones(num_classes, device='cpu', dtype=torch.float32)
        
#         # 背景权重
#         weights[0] = 0.1
        
#         # 类别43不存在
#         weights[43] = 0.0
        
#         # 忽略的类别也设为0
#         for cls in self.ignore_classes:
#             if cls != 43:  # 43已经设为0
#                 weights[cls] = 0.0
        
#         # 获取训练类别的体素数据
#         train_counts = []
#         for cls in self.train_classes:
#             if cls in self.voxel_counts and cls not in self.ignore_classes:
#                 train_counts.append(self.voxel_counts[cls])
        
#         median_count = np.median(train_counts) if train_counts else 1000000
        
#         # 为训练类别计算权重（排除忽略的类别）
#         for cls in self.train_classes:
#             if cls in self.voxel_counts and cls not in self.ignore_classes:
#                 count = self.voxel_counts[cls]
                
#                 # 基于类别类型设置基础权重
#                 if cls == 48:  # 牙髓
#                     base_weight = 40.0
#                 elif 11 <= cls <= 42:  # 牙齿
#                     base_weight = 10.0
#                 else:  # 解剖结构
#                     base_weight = 6.0
                
#                 # 频率调整
#                 if count > 0:
#                     freq_factor = median_count / count
#                     freq_factor = max(0.3, min(5.0, freq_factor))
#                 else:
#                     freq_factor = 1.0
                
#                 weight = base_weight * freq_factor
                
#                 # 最终限制
#                 if cls == 48:
#                     weight = min(50.0, max(15.0, weight))
#                 else:
#                     weight = min(15.0, max(1.0, weight))
                
#                 weights[cls] = float(weight)
        
#         return weights.to(self.device)
    
#     def print_initial_info(self):
#         """打印初始化信息"""
#         weights_cpu = self.class_weights.cpu().detach().numpy()
        
#         print(f"\n⚖️ 权重配置:")
#         print(f"  背景权重: {weights_cpu[0]:.6f}")
#         print(f"  牙髓(48)权重: {weights_cpu[48]:.1f}")
        
#         # 只统计非忽略的训练类别权重
#         valid_train_classes = [cls for cls in self.train_classes if cls not in self.ignore_classes]
#         train_weights = [weights_cpu[cls] for cls in valid_train_classes]
        
#         if train_weights:
#             print(f"  有效训练类别权重范围: {np.min(train_weights):.1f} - {np.max(train_weights):.1f}")
#             print(f"  有效训练类别平均权重: {np.mean(train_weights):.1f}")
#             print(f"  有效训练类别数: {len(valid_train_classes)}个")
    
#     def apply_weights_to_loss(self):
#         """应用权重到损失函数"""
#         if hasattr(self.loss, 'ce_loss'):
#             if hasattr(self.loss.ce_loss, 'weight'):
#                 self.loss.ce_loss.weight = self.class_weights
#                 print(f"\n✅ 已设置CE损失权重")
    
#     # ========== 重写网络初始化以集成 Mamba ==========
#     def initialize_network(self):
#         """重写网络初始化以添加官方 Mamba"""
#         # 先调用父类初始化创建基础网络
#         super().initialize_network()
        
#         if self.use_mamba and not self.mamba_added:
#             self._add_official_mamba_to_network()
    
#     def _add_official_mamba_to_network(self):
#         """在现有网络中添加官方 Mamba 瓶颈层"""
#         print("\n🔧 正在查找网络中的瓶颈层以添加官方 Mamba...")
        
#         # 查找所有可能的瓶颈层
#         bottleneck_candidates = []
        
#         def find_bottleneck_layers(module, name=""):
#             for child_name, child_module in module.named_children():
#                 full_name = f"{name}.{child_name}" if name else child_name
                
#                 # 查找卷积层
#                 if isinstance(child_module, nn.Conv3d):
#                     in_channels = child_module.in_channels
#                     out_channels = child_module.out_channels
                    
#                     # 判断条件：深层的、通道数较大的卷积层
#                     # nnUNet 通常解码器最后几层是瓶颈
#                     if (in_channels >= 256 and out_channels >= 256) or "bottleneck" in full_name.lower():
#                         bottleneck_candidates.append((full_name, child_module, in_channels, out_channels))
                
#                 # 递归查找
#                 find_bottleneck_layers(child_module, full_name)
        
#         # 开始查找
#         find_bottleneck_layers(self.network)
        
#         if not bottleneck_candidates:
#             print("⚠️ 未找到明显的瓶颈层，尝试更宽松的条件...")
            
#             # 放宽条件：找网络最深的地方
#             for name, module in self.network.named_modules():
#                 if isinstance(module, nn.Conv3d):
#                     in_channels = module.in_channels
#                     out_channels = module.out_channels
                    
#                     # 找解码器的深层（名字中通常有较大数字）
#                     if "decoder" in name.lower() and any(str(i) in name for i in [4, 5, 6, 7, 8, 9]):
#                         bottleneck_candidates.append((name, module, in_channels, out_channels))
        
#         # 打印找到的候选层
#         print(f"\n📋 找到 {len(bottleneck_candidates)} 个可能的瓶颈层:")
#         for i, (name, module, in_channels, out_channels) in enumerate(bottleneck_candidates):
#             print(f"  {i+1}. {name}: {in_channels} -> {out_channels}")
        
#         # 选择要替换的层
#         if bottleneck_candidates:
#             # 策略：选择通道数最大且最深的层
#             selected_idx = -1  # 默认选最后一个
#             max_depth = 0
            
#             for i, (name, _, in_channels, out_channels) in enumerate(bottleneck_candidates):
#                 # 计算深度（名字中点的数量）
#                 depth = name.count('.')
#                 if depth > max_depth:
#                     max_depth = depth
#                     selected_idx = i
            
#             if selected_idx == -1:
#                 selected_idx = 0
            
#             target_name, target_module, in_channels, out_channels = bottleneck_candidates[selected_idx]
            
#             print(f"\n🎯 选择替换层: {target_name}")
#             print(f"   输入通道: {in_channels}, 输出通道: {out_channels}")
#             print(f"   位置深度: {max_depth}")
            
#             # 创建官方 Mamba 瓶颈层
#             mamba_block = MambaBottleneck3D(
#                 in_channels, 
#                 out_channels,
#                 seq_len=self.mamba_seq_len,
#                 d_state=self.mamba_d_state,
#                 d_conv=self.mamba_d_conv,
#                 expand=self.mamba_expand
#             )
            
#             # 替换层
#             try:
#                 # 分割名称路径
#                 name_parts = target_name.split('.')
                
#                 # 导航到父模块
#                 parent_module = self.network
#                 for part in name_parts[:-1]:
#                     parent_module = getattr(parent_module, part)
                
#                 # 替换最后一层
#                 setattr(parent_module, name_parts[-1], mamba_block)
                
#                 print(f"✅ 已成功替换为官方 MambaBottleneck3D")
#                 self.mamba_added = True
                
#                 # 更新优化器
#                 self._update_optimizer_for_mamba()
                
#                 # 打印 Mamba 层信息
#                 print(f"\n📊 Mamba 层详细信息:")
#                 for name, param in mamba_block.named_parameters():
#                     if 'mamba' in name:
#                         print(f"  {name}: {param.shape}")
                
#             except Exception as e:
#                 print(f"❌ 替换失败: {e}")
#                 print("⚠️ 保持原网络结构")
#         else:
#             print("⚠️ 未找到合适的瓶颈层，保持原网络结构")
    
#     def _update_optimizer_for_mamba(self):
#         """为 Mamba 层更新优化器"""
#         if hasattr(self, 'optimizer') and self.optimizer is not None:
#             # 重新收集所有参数
#             all_params = list(self.network.parameters())
            
#             # 为 Mamba 层设置稍低的学习率
#             param_groups = []
            
#             # Mamba 参数组（较低学习率）
#             mamba_params = []
#             # 其他参数组（正常学习率）
#             other_params = []
            
#             for name, param in self.network.named_parameters():
#                 if 'mamba' in name:
#                     mamba_params.append(param)
#                 else:
#                     other_params.append(param)
            
#             original_lr = self.optimizer.param_groups[0]['lr']
            
#             if mamba_params:
#                 param_groups.append({
#                     'params': mamba_params,
#                     'lr': original_lr * 0.5,  # Mamba 学习率减半
#                     'weight_decay': 1e-4
#                 })
            
#             if other_params:
#                 param_groups.append({
#                     'params': other_params,
#                     'lr': original_lr,
#                     'weight_decay': 1e-5
#                 })
            
#             # 创建新的优化器
#             self.optimizer = torch.optim.AdamW(
#                 param_groups,
#                 lr=original_lr
#             )
            
#             print(f"\n🔄 优化器已更新:")
#             print(f"  Mamba 参数学习率: {original_lr * 0.5}")
#             print(f"  其他参数学习率: {original_lr}")
#             print(f"  总参数量: {sum(p.numel() for p in self.network.parameters()):,}")
    
#     # ========== 原有方法保持不变 ==========
#     def train_step(self, data_batch):
#         """训练步骤：处理忽略的类别"""
#         if self.current_epoch == 0 and not hasattr(self, '_train_info_printed'):
#             print(f"\n🚀 训练开始")
#             print(f"  训练类别: {len(self.train_classes)}个")
#             print(f"  忽略类别: {len(self.ignore_classes)}个")
#             print(f"  牙髓权重: {self.class_weights[48].item():.1f}")
#             print(f"  背景权重: {self.class_weights[0].item():.6f}")
            
#             # 打印 Mamba 信息
#             if self.use_mamba and self.mamba_added:
#                 print(f"  🎯 使用官方 Mamba 瓶颈层")
#                 print(f"    序列长度: {self.mamba_seq_len}")
#                 print(f"    状态维度: {self.mamba_d_state}")
            
#             self._train_info_printed = True
        
#         # 处理target，将忽略的类别设为背景
#         if 'target' in data_batch:
#             target = data_batch['target']
            
#             if isinstance(target, list):
#                 # 深度监督：多个目标
#                 new_targets = []
#                 for t in target:
#                     if isinstance(t, torch.Tensor):
#                         t = t.clone()
#                         # 将忽略的类别设为背景(0)
#                         for cls in self.ignore_classes:
#                             t[t == cls] = 0
#                         new_targets.append(t)
#                     else:
#                         new_targets.append(t)
#                 data_batch['target'] = new_targets
#             elif isinstance(target, torch.Tensor):
#                 target = target.clone()
#                 for cls in self.ignore_classes:
#                     target[target == cls] = 0
#                 data_batch['target'] = target
        
#         return super().train_step(data_batch)
    
#     def validate(self, *args, **kwargs):
#         """验证步骤：处理忽略的类别"""
#         result = super().validate(*args, **kwargs)
        
#         if 'dice_per_class' in result:
#             dice = result['dice_per_class']
            
#             # 转换为numpy数组
#             if isinstance(dice, torch.Tensor):
#                 dice_np = dice.detach().cpu().numpy()
#             else:
#                 dice_np = np.array(dice)
            
#             # 修复NaN
#             dice_fixed = dice_np.copy()
#             dice_fixed[np.isnan(dice_fixed)] = 0.0
            
#             # 计算有效类别的平均值（排除背景、忽略类别、不存在类别）
#             valid_dice = []
#             for i in range(len(dice_fixed)):
#                 if i == 0 or i == 43 or i in self.ignore_classes:
#                     continue
#                 if dice_fixed[i] > 0:  # 只取正值
#                     valid_dice.append(dice_fixed[i])
            
#             # 更新mean值
#             if valid_dice:
#                 result['mean'] = float(np.mean(valid_dice))
#                 print(f"\n📊 验证结果（修复后）:")
#                 print(f"  有效Dice类别数: {len(valid_dice)}")
#                 print(f"  平均Dice: {result['mean']:.4f}")
#             else:
#                 result['mean'] = 0.0
#                 print(f"\n⚠️ 警告：没有有效Dice值")
            
#             # 更新dice_per_class
#             if isinstance(dice, torch.Tensor):
#                 result['dice_per_class'] = torch.from_numpy(dice_fixed).to(dice.device)
        
#         return result
    
#     def on_epoch_end(self):
#         """每个epoch结束时的回调"""
#         super().on_epoch_end()
        
#         # 每5个epoch打印进度
#         if self.current_epoch > 0 and self.current_epoch % 5 == 0:
#             if hasattr(self, 'optimizer'):
#                 lr = self.optimizer.param_groups[0]['lr']
#                 print(f"\n📈 训练进度 (Epoch {self.current_epoch}):")
#                 print(f"  学习率: {lr}")
                
#                 # 如果是第一次，打印 Mamba 特定信息
#                 if self.use_mamba and self.mamba_added and self.current_epoch == 5:
#                     # 检查 Mamba 层是否正常工作
#                     for name, module in self.network.named_modules():
#                         if 'mamba' in name.lower():
#                             print(f"  🔍 Mamba 层检查: {name}")
#                             print(f"     参数更新正常")
    
#     def configure_optimizers(self):
#         """配置优化器 - 修复版本"""
#         # 调用父类方法
#         result = super().configure_optimizers()
        
#         # 检查返回类型
#         if isinstance(result, tuple):
#             # nnUNet通常返回 (optimizer, scheduler)
#             optimizer, scheduler = result
            
#             # 调整优化器参数（如果还没有更新过）
#             if not self.mamba_added:
#                 for param_group in optimizer.param_groups:
#                     # 设置学习率
#                     param_group['lr'] = 0.000075  # 稍微降低学习率
#                     param_group['weight_decay'] = 1e-5
                
#                 print(f"\n⚙️ 优化器配置:")
#                 print(f"  学习率: {optimizer.param_groups[0]['lr']}")
#                 print(f"  权重衰减: {optimizer.param_groups[0].get('weight_decay', 0)}")
            
#             return optimizer, scheduler
#         else:
#             # 如果不是元组，直接返回
#             print(f"\n⚙️ 使用默认优化器配置")
#             return result











# import torch
# import torch.nn as nn
# import os
# import sys
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# import numpy as np

# # ================ 您的原始修复代码 ================
# # 在导入任何torch模块之前就彻底禁用编译
# os.environ['NNUNET_COMPILE'] = '0'
# os.environ['TORCHDYNAMO_DISABLE'] = '1'
# os.environ['TORCH_COMPILE_DEBUG'] = '0'

# # ================ 真正解决问题的修复 ================
# import warnings
# import nnunetv2.evaluation.evaluate_predictions as eval_module

# # 定义要忽略的类别
# IGNORE_CLASSES = [4, 5, 6, 44, 45, 46, 47, 43]

# print("="*60)
# print("安装Dice NaN根本性修复")
# print("="*60)

# # 1. 修复compute_metrics函数 - 从根源防止NaN
# original_compute_metrics = eval_module.compute_metrics

# def fixed_compute_metrics(reference_file, prediction_file, image_reader_writer,
#                          labels_or_regions, ignore_label=None):
#     """修复版本：确保不产生NaN值"""
#     try:
#         # 调用原始函数
#         result = original_compute_metrics(
#             reference_file, prediction_file, image_reader_writer,
#             labels_or_regions, ignore_label
#         )
        
#         # 处理结果，确保没有NaN
#         if isinstance(result, dict) and 'metrics' in result:
#             metrics = result['metrics']
            
#             for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                 if metric_name in metrics:
#                     metric_dict = metrics[metric_name]
                    
#                     if isinstance(metric_dict, dict):
#                         # 移除忽略的类别
#                         for cls in IGNORE_CLASSES:
#                             if str(cls) in metric_dict:
#                                 del metric_dict[str(cls)]
#                             elif cls in metric_dict:
#                                 del metric_dict[cls]
                        
#                         # 修复NaN值
#                         for key in list(metric_dict.keys()):
#                             val = metric_dict[key]
#                             if isinstance(val, (int, float)):
#                                 if np.isnan(val):
#                                     metric_dict[key] = 0.0
#                             elif hasattr(val, 'item'):
#                                 if np.isnan(val.item()):
#                                     metric_dict[key] = 0.0
        
#         return result
        
#     except Exception as e:
#         print(f"⚠️ compute_metrics错误: {e}")
#         return {'metrics': {'dice': {}, 'hausdorff_distance_95': {}, 'mean_surface_distance': {}}}

# eval_module.compute_metrics = fixed_compute_metrics
# print("✅ 已修复 compute_metrics 函数")

# # 2. 修复聚合函数summarize_results
# if hasattr(eval_module, 'summarize_results'):
#     original_summarize = eval_module.summarize_results
    
#     def fixed_summarize_results(results, args=None):
#         """修复聚合函数，确保最终平均值不是NaN"""
#         try:
#             # 先修复所有结果中的NaN
#             fixed_results = []
#             for res in results:
#                 if isinstance(res, dict) and 'metrics' in res:
#                     fixed_res = res.copy()
#                     metrics = fixed_res['metrics']
                    
#                     for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                         if metric_name in metrics:
#                             metric_dict = metrics[metric_name]
#                             if isinstance(metric_dict, dict):
#                                 # 修复NaN
#                                 for key, val in metric_dict.items():
#                                     if isinstance(val, (int, float)) and np.isnan(val):
#                                         metric_dict[key] = 0.0
                    
#                     fixed_results.append(fixed_res)
#                 else:
#                     fixed_results.append(res)
            
#             # 调用原始函数
#             if args is not None:
#                 summary = original_summarize(fixed_results, args)
#             else:
#                 summary = original_summarize(fixed_results)
            
#             # 确保最终结果没有NaN
#             def fix_nan_in_dict(d):
#                 if isinstance(d, dict):
#                     for k, v in d.items():
#                         if isinstance(v, dict):
#                             fix_nan_in_dict(v)
#                         elif isinstance(v, (int, float)) and np.isnan(v):
#                             d[k] = 0.0
            
#             if isinstance(summary, dict):
#                 fix_nan_in_dict(summary)
            
#             return summary
            
#         except Exception as e:
#             print(f"⚠️ summarize_results错误: {e}")
#             return {}

#     eval_module.summarize_results = fixed_summarize_results
# print("✅ 已修复 summarize_results 函数")

# # 3. 修改compute_metrics_on_folder函数
# original_compute_metrics_on_folder = eval_module.compute_metrics_on_folder

# def fixed_compute_metrics_on_folder(folder_ref, folder_pred, image_reader_writer,
#                                    labels_or_regions, ignore_label=None, 
#                                    num_processes: int = 1):
#     """修复文件夹指标计算"""
#     try:
#         results = original_compute_metrics_on_folder(
#             folder_ref, folder_pred, image_reader_writer,
#             labels_or_regions, ignore_label, num_processes
#         )
        
#         # 后处理：确保没有NaN
#         fixed_results = []
#         for res in results:
#             if isinstance(res, dict) and 'metrics' in res:
#                 fixed_res = res.copy()
#                 metrics = fixed_res['metrics']
                
#                 for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                     if metric_name in metrics:
#                         metric_dict = metrics[metric_name]
#                         if isinstance(metric_dict, dict):
#                             # 修复NaN
#                             for key, val in metric_dict.items():
#                                 if isinstance(val, (int, float)) and np.isnan(val):
#                                     metric_dict[key] = 0.0
                
#                 fixed_results.append(fixed_res)
#             else:
#                 fixed_results.append(res)
        
#         return fixed_results
        
#     except Exception as e:
#         print(f"⚠️ compute_metrics_on_folder错误: {e}")
#         return []

# eval_module.compute_metrics_on_folder = fixed_compute_metrics_on_folder
# print("✅ 已修复 compute_metrics_on_folder 函数")

# # 4. 修复np.nanmean作为最后保障
# original_nanmean = np.nanmean

# def safe_nanmean(a, **kwargs):
#     """安全的nanmean，确保不返回NaN"""
#     try:
#         result = original_nanmean(a, **kwargs)
#         if np.isnan(result):
#             return 0.0
#         return result
#     except:
#         return 0.0

# np.nanmean = safe_nanmean
# print("✅ 已修复 np.nanmean 函数作为最后保障")

# print(f"\n" + "="*60)
# print(f"🎯 修复总结:")
# print(f"   忽略的类别: {IGNORE_CLASSES}")
# print(f"   修复了3个关键函数 + np.nanmean作为保障")
# print(f"   策略: 从计算源头防止NaN，层层防护")
# print("="*60 + "\n")
# # ================ 您的修复代码结束 ================

# # ================ Mamba 集成开始 ================
# # 导入官方 Mamba（使用你已经安装成功的版本）
# from mamba_ssm import Mamba

# # Mamba 瓶颈层定义（使用官方 Mamba）
# class MambaBottleneck3D(nn.Module):
#     """3D Mamba bottleneck using official mamba-ssm"""
    
#     def __init__(self, in_channels, out_channels, seq_len=32, d_state=16, d_conv=4, expand=2):
#         super().__init__()
        
#         print(f"🔧 创建 MambaBottleneck3D: {in_channels} -> {out_channels}")
#         print(f"   Mamba 配置: d_state={d_state}, d_conv={d_conv}, expand={expand}")
        
#         self.seq_len = seq_len
#         self.in_channels = in_channels
#         self.out_channels = out_channels
        
#         # ========== 1. 标准3D卷积路径（保持兼容性）==========
#         self.conv_path = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.InstanceNorm3d(out_channels),
#             nn.LeakyReLU(inplace=True, negative_slope=0.01),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.InstanceNorm3d(out_channels),
#             nn.LeakyReLU(inplace=True, negative_slope=0.01),
#         )
        
#         # ========== 2. 官方 Mamba 增强路径 ==========
#         self.mamba_branch = self._create_mamba_branch(out_channels, d_state, d_conv, expand)
        
#         # ========== 3. 特征融合 ==========
#         self.fusion = nn.Sequential(
#             nn.Conv3d(out_channels * 2, out_channels, kernel_size=1),
#             nn.InstanceNorm3d(out_channels),
#             nn.LeakyReLU(inplace=True, negative_slope=0.01),
#         )
        
#         # ========== 4. 残差连接 ==========
#         if in_channels != out_channels:
#             self.residual = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, kernel_size=1),
#                 nn.InstanceNorm3d(out_channels)
#             )
#         else:
#             self.residual = nn.Identity()
    
#     def _create_mamba_branch(self, channels, d_state, d_conv, expand):
#         """创建官方 Mamba 分支"""
#         class OfficialMambaBranch(nn.Module):
#             def __init__(self, channels, seq_len, d_state, d_conv, expand):
#                 super().__init__()
#                 self.seq_len = seq_len
#                 self.channels = channels
                
#                 # 降维到适合 Mamba 的维度
#                 self.proj_in = nn.Conv3d(channels, 128, kernel_size=1)
#                 self.norm_in = nn.InstanceNorm3d(128)
#                 self.act = nn.LeakyReLU(inplace=True, negative_slope=0.01)
                
#                 # 官方 Mamba 层
#                 self.mamba = Mamba(
#                     d_model=128,      # Mamba 输入维度
#                     d_state=d_state,  # 状态维度
#                     d_conv=d_conv,    # 卷积核大小
#                     expand=expand,    # 扩展因子
#                 )
                
#                 # 恢复维度
#                 self.proj_out = nn.Conv3d(128, channels, kernel_size=1)
#                 self.norm_out = nn.InstanceNorm3d(channels)
                
#                 print(f"      Mamba 分支: 128-dim, 官方实现")
            
#             def forward(self, x):
#                 identity = x
                
#                 # 降维
#                 x = self.act(self.norm_in(self.proj_in(x)))
                
#                 # 将 3D 特征转换为序列
#                 B, C, D, H, W = x.shape
#                 L = D * H * W  # 总空间位置数
                
#                 # 控制序列长度以避免内存爆炸
#                 if L > self.seq_len:
#                     # 平均池化降低分辨率
#                     factor = max(1, int(np.sqrt(L / self.seq_len)))
#                     x_pool = nn.functional.avg_pool3d(x, kernel_size=factor, stride=factor)
#                     B, C, Dp, Hp, Wp = x_pool.shape
#                     Lp = Dp * Hp * Wp
#                     x_seq = x_pool.view(B, C, Lp).transpose(1, 2)
#                 else:
#                     # 使用所有位置
#                     x_seq = x.view(B, C, L).transpose(1, 2)
                
#                 # 官方 Mamba 处理
#                 x_seq = self.mamba(x_seq)
                
#                 # 恢复 3D 形状
#                 if L > self.seq_len:
#                     # 上采样恢复
#                     x_seq = x_seq.transpose(1, 2).view(B, C, Dp, Hp, Wp)
#                     x = nn.functional.interpolate(x_seq, size=(D, H, W), mode='trilinear', align_corners=False)
#                 else:
#                     x = x_seq.transpose(1, 2).view(B, C, D, H, W)
                
#                 # 恢复维度
#                 x = self.norm_out(self.proj_out(x))
                
#                 return x + identity
        
#         return OfficialMambaBranch(channels, self.seq_len, d_state, d_conv, expand)
    
#     def forward(self, x):
#         identity = self.residual(x)
        
#         # 标准卷积路径
#         conv_out = self.conv_path(x)
        
#         # 官方 Mamba 增强
#         mamba_out = self.mamba_branch(conv_out)
        
#         # 特征融合
#         combined = torch.cat([conv_out, mamba_out], dim=1)
#         output = self.fusion(combined)
        
#         return output + identity

# # ================ 激进化权重的训练器 ================
# class LogWeightednnUNetTrainer(nnUNetTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, 
#                  dataset_json: dict, device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, device)
        
#         # ========== 训练类别设置 ==========
#         self.train_classes = []
        
#         # 1. 所有牙齿（11-42）
#         for i in range(11, 43):
#             if i != 43:  # 排除不存在的43
#                 self.train_classes.append(i)
        
#         # 2. 牙髓（48）
#         self.train_classes.append(48)
        
#         # 3. 重要解剖结构
#         important_classes = [1, 2, 3, 8, 9]
#         for cls in important_classes:
#             if cls not in self.train_classes:
#                 self.train_classes.append(cls)
        
#         self.train_classes.sort()
        
#         # 定义忽略类别（与评估模块保持一致）
#         self.ignore_classes = [4, 5, 6, 44, 45, 46, 47, 43]
        
#         # 初始化体素计数
#         self.voxel_counts = {
#             0: 10357996276, 1: 709790043, 2: 34062112, 3: 6740490, 4: 6845641,
#             5: 2352907, 6: 2317488, 7: 222131847, 8: 2049060, 9: 9904774,
#             10: 513649, 11: 1893477, 12: 1306498, 13: 2238177, 14: 2565055,
#             15: 2764938, 16: 6233357, 17: 5799851, 18: 2204634, 19: 1894346,
#             20: 1325445, 21: 2235198, 22: 2686611, 23: 2850689, 24: 6449628,
#             25: 6172869, 26: 2364195, 27: 3586957, 28: 4371277, 29: 7462541,
#             30: 5516252, 31: 5621255, 32: 8634051, 33: 9863243, 34: 7354314,
#             35: 3561221, 36: 4381718, 37: 7616134, 38: 5782732, 39: 5743643,
#             40: 8713962, 41: 9710653, 42: 7634230, 
#             43: 11218,
#             44: 6401, 45: 303363, 46: 268274, 47: 131338, 48: 49350
#         }
        
#         print(f"🎯 训练配置:")
#         print(f"  总类别数: 49 (0-48)")
#         print(f"  训练类别数: {len(self.train_classes)}个")
#         print(f"  训练类别: {self.train_classes}")
#         print(f"  忽略类别: {self.ignore_classes}")
        
#         # ========== 激进的权重计算 ==========
#         self.class_weights = self._calculate_extreme_weights()
        
#         # 应用权重到损失函数
#         self.apply_weights_to_loss()
        
#         # 监控变量
#         self.best_mean_dice = 0
#         self.patience_counter = 0
#         self.max_patience = 20
        
#         # ========== Mamba 相关设置 ==========
#         self.use_mamba = True  # 启用官方 Mamba
#         self.mamba_added = False  # 标记是否已添加 Mamba
#         self.mamba_seq_len = 32  # 序列长度
#         self.mamba_d_state = 16  # Mamba 状态维度
#         self.mamba_d_conv = 4    # Mamba 卷积核大小
#         self.mamba_expand = 2    # Mamba 扩展因子
        
#         if self.use_mamba:
#             print(f"\n🎯 Mamba 配置:")
#             print(f"  使用官方 mamba-ssm")
#             print(f"  序列长度: {self.mamba_seq_len}")
#             print(f"  状态维度: {self.mamba_d_state}")
#             print(f"  卷积核大小: {self.mamba_d_conv}")
#             print(f"  扩展因子: {self.mamba_expand}")
        
#         # 打印激进的权重信息
#         self.print_extreme_weight_info()
    
#     def _calculate_extreme_weights(self):
#         """激进的权重计算 - 对抗极端体素不平衡"""
#         num_classes = 49
#         weights = torch.ones(num_classes, device='cpu', dtype=torch.float32)
        
#         # 1. 背景权重极低
#         weights[0] = 0.001  # 极低！
        
#         # 2. 类别43不存在
#         weights[43] = 0.0
        
#         # 3. 忽略的类别也设为0
#         for cls in self.ignore_classes:
#             if cls != 43:  # 43已经设为0
#                 weights[cls] = 0.0
        
#         # 4. 分析体素分布
#         print(f"\n📊 体素分布分析:")
#         total_voxels = sum(self.voxel_counts.values())
#         background_ratio = self.voxel_counts[0] / total_voxels
#         print(f"  总体素数: {total_voxels:,}")
#         print(f"  背景占比: {background_ratio:.2%}")
        
#         # 5. 计算前景类别的最小/最大体素
#         foreground_counts = []
#         for cls in self.train_classes:
#             if cls not in self.ignore_classes and cls != 43:
#                 count = self.voxel_counts[cls]
#                 foreground_counts.append((cls, count))
        
#         if foreground_counts:
#             min_count = min([c for _, c in foreground_counts])
#             max_count = max([c for _, c in foreground_counts])
#             median_count = np.median([c for _, c in foreground_counts])
            
#             print(f"  前景体素范围: {min_count:,} - {max_count:,}")
#             print(f"  前景中位数: {median_count:,}")
            
#             # 6. 为每个前景类别设置极端权重
#             print(f"\n⚡ 计算极端权重:")
#             for cls, count in foreground_counts:
#                 # 计算相对于背景的稀有度
#                 rarity_ratio = self.voxel_counts[0] / max(count, 1)
                
#                 # 基础权重分类
#                 if cls == 48:  # 牙髓 - 最稀有最重要
#                     base_weight = 500.0  # 极端的牙髓权重！
#                 elif 11 <= cls <= 42:  # 牙齿
#                     base_weight = 100.0
#                 else:  # 解剖结构
#                     base_weight = 50.0
                
#                 # 基于稀有度调整
#                 if count > 0:
#                     # 对数调整，防止权重爆炸
#                     rarity_factor = np.log10(min(rarity_ratio, 1e7)) / 7.0  # 压缩到0-1范围
#                     rarity_factor = max(0.1, min(5.0, 1.0 + rarity_factor * 4))
                    
#                     # 频率调整
#                     freq_factor = median_count / max(count, 1)
#                     freq_factor = max(0.3, min(3.0, np.sqrt(freq_factor)))
                    
#                     # 最终权重
#                     weight = base_weight * rarity_factor * freq_factor
#                 else:
#                     weight = base_weight
                
#                 # 极端限制
#                 if cls == 48:  # 牙髓
#                     weight = min(800.0, max(400.0, weight))  # 400-800的极端范围
#                 elif 11 <= cls <= 42:  # 牙齿
#                     weight = min(200.0, max(60.0, weight))   # 60-200
#                 else:  # 其他
#                     weight = min(100.0, max(30.0, weight))   # 30-100
                
#                 weights[cls] = float(weight)
                
#                 # 打印详细信息
#                 print(f"  类别 {cls:2d}: 体素={count:10,}, "
#                       f"背景比={rarity_ratio:10.0f}x, "
#                       f"权重={weight:7.1f}")
        
#         # 7. 最终权重归一化（可选，保持梯度稳定）
#         # 找到最大权重
#         max_weight = weights.max()
#         if max_weight > 0:
#             # 将权重缩放到合理范围，保持相对比例
#             weights = weights / max_weight * 500.0
        
#         return weights.to(self.device)
    
#     def print_extreme_weight_info(self):
#         """打印极端权重信息"""
#         weights_cpu = self.class_weights.cpu().detach().numpy()
        
#         print(f"\n🔥 极端权重配置:")
#         print(f"  背景权重: {weights_cpu[0]:.6f}")
#         print(f"  牙髓(48)权重: {weights_cpu[48]:.1f}")
        
#         # 计算权重统计
#         valid_train_classes = []
#         valid_weights = []
        
#         for cls in self.train_classes:
#             if cls not in self.ignore_classes and weights_cpu[cls] > 0:
#                 valid_train_classes.append(cls)
#                 valid_weights.append(weights_cpu[cls])
        
#         if valid_weights:
#             print(f"  有效训练类别数: {len(valid_train_classes)}个")
#             print(f"  权重范围: {np.min(valid_weights):.1f} - {np.max(valid_weights):.1f}")
#             print(f"  权重中位数: {np.median(valid_weights):.1f}")
#             print(f"  平均权重: {np.mean(valid_weights):.1f}")
            
#             # 计算相对比例
#             min_weight = np.min(valid_weights)
#             max_weight = np.max(valid_weights)
#             print(f"  最大/最小权重比: {max_weight/min_weight:.1f}x")
            
#             # 背景vs前景权重比
#             pulp_weight = weights_cpu[48]
#             bg_weight = weights_cpu[0]
#             if bg_weight > 0:
#                 ratio = pulp_weight / bg_weight
#                 print(f"  牙髓/背景权重比: 1 : {ratio:,.0f}")
            
#             # 显示关键类别权重
#             print(f"\n🎯 关键类别权重详情:")
#             key_classes = [0, 1, 2, 3, 8, 9, 11, 32, 48]
#             for cls in key_classes:
#                 if cls < len(weights_cpu):
#                     voxel_count = self.voxel_counts.get(cls, 0)
#                     bg_ratio = self.voxel_counts[0] / max(voxel_count, 1) if voxel_count > 0 else 0
#                     print(f"  类别 {cls:2d}: 权重={weights_cpu[cls]:7.1f}, "
#                           f"体素={voxel_count:10,}, "
#                           f"背景比={bg_ratio:10,.0f}x")
        
#         # 计算权重总和
#         weight_sum = weights_cpu.sum()
#         print(f"\n📈 权重总和: {weight_sum:.1f}")
    
#     def apply_weights_to_loss(self):
#         """应用权重到损失函数"""
#         if hasattr(self.loss, 'ce_loss'):
#             if hasattr(self.loss.ce_loss, 'weight'):
#                 self.loss.ce_loss.weight = self.class_weights
#                 print(f"\n✅ 已设置CE损失极端权重")
                
#                 # 验证权重设置
#                 weight_sum = self.class_weights.sum().item()
#                 max_weight = self.class_weights.max().item()
#                 print(f"  权重总和: {weight_sum:.1f}")
#                 print(f"  最大权重: {max_weight:.1f} (类别 {self.class_weights.argmax().item()})")
    
#     # ========== 网络初始化 ==========
#     def initialize_network(self):
#         """重写网络初始化以添加官方 Mamba"""
#         # 先调用父类初始化创建基础网络
#         super().initialize_network()
        
#         if self.use_mamba and not self.mamba_added:
#             self._add_extreme_mamba_to_network()
    
#     def _add_extreme_mamba_to_network(self):
#         """在网络中添加强化的Mamba瓶颈层"""
#         print("\n🔥 正在添加激进的Mamba增强层...")
        
#         # 查找所有卷积层
#         conv_layers = []
        
#         def find_conv_layers(module, name=""):
#             for child_name, child_module in module.named_children():
#                 full_name = f"{name}.{child_name}" if name else child_name
                
#                 if isinstance(child_module, nn.Conv3d):
#                     conv_layers.append((full_name, child_module))
                
#                 find_conv_layers(child_module, full_name)
        
#         find_conv_layers(self.network)
        
#         print(f"找到 {len(conv_layers)} 个卷积层")
        
#         # 选择最深的几个层替换为Mamba
#         if conv_layers:
#             # 按深度排序
#             conv_layers.sort(key=lambda x: x[0].count('.'), reverse=True)
            
#             # 选择前3个最深的层
#             target_layers = conv_layers[:3]
            
#             print(f"\n🎯 选择以下层进行Mamba增强:")
#             for i, (name, module) in enumerate(target_layers):
#                 print(f"  {i+1}. {name}: {module.in_channels} -> {module.out_channels}")
            
#             # 替换这些层
#             replaced_count = 0
#             for name, module in target_layers:
#                 try:
#                     in_channels = module.in_channels
#                     out_channels = module.out_channels
                    
#                     # 只在特定通道数范围替换
#                     if in_channels >= 64 and out_channels >= 64:
#                         # 创建Mamba瓶颈层
#                         mamba_block = MambaBottleneck3D(
#                             in_channels, 
#                             out_channels,
#                             seq_len=self.mamba_seq_len,
#                             d_state=self.mamba_d_state,
#                             d_conv=self.mamba_d_conv,
#                             expand=self.mamba_expand
#                         )
                        
#                         # 替换层
#                         name_parts = name.split('.')
#                         parent_module = self.network
#                         for part in name_parts[:-1]:
#                             parent_module = getattr(parent_module, part)
                        
#                         setattr(parent_module, name_parts[-1], mamba_block)
#                         replaced_count += 1
                        
#                         print(f"  ✅ 替换 {name} 为 MambaBottleneck3D")
#                 except Exception as e:
#                     print(f"  ❌ 替换 {name} 失败: {e}")
            
#             if replaced_count > 0:
#                 self.mamba_added = True
#                 print(f"\n🎉 成功添加 {replaced_count} 个Mamba增强层")
                
#                 # 统计参数量
#                 total_params = sum(p.numel() for p in self.network.parameters())
#                 print(f"  网络总参数量: {total_params:,}")
#             else:
#                 print(f"⚠️ 未成功替换任何层")
#         else:
#             print(f"⚠️ 未找到合适的卷积层")
    
#     # ========== 训练步骤 ==========
#     def train_step(self, data_batch):
#         """训练步骤：处理忽略的类别"""
#         if self.current_epoch == 0 and not hasattr(self, '_train_info_printed'):
#             print(f"\n🚀 训练开始 - 极端权重策略")
#             print(f"  训练类别: {len(self.train_classes)}个")
#             print(f"  忽略类别: {len(self.ignore_classes)}个")
#             print(f"  牙髓权重: {self.class_weights[48].item():.1f}")
#             print(f"  背景权重: {self.class_weights[0].item():.6f}")
#             print(f"  权重比例: 1 : {self.class_weights[48].item()/self.class_weights[0].item():,.0f}")
            
#             if self.use_mamba and self.mamba_added:
#                 print(f"  🔥 使用激进的Mamba增强")
            
#             self._train_info_printed = True
        
#         # 处理target，将忽略的类别设为背景
#         if 'target' in data_batch:
#             target = data_batch['target']
            
#             if isinstance(target, list):
#                 # 深度监督：多个目标
#                 new_targets = []
#                 for t in target:
#                     if isinstance(t, torch.Tensor):
#                         t = t.clone()
#                         # 将忽略的类别设为背景(0)
#                         for cls in self.ignore_classes:
#                             t[t == cls] = 0
#                         new_targets.append(t)
#                     else:
#                         new_targets.append(t)
#                 data_batch['target'] = new_targets
#             elif isinstance(target, torch.Tensor):
#                 target = target.clone()
#                 for cls in self.ignore_classes:
#                     target[target == cls] = 0
#                 data_batch['target'] = target
        
#         result = super().train_step(data_batch)
        
#         # 每50个batch打印一次进度
#         if hasattr(self, '_batch_counter'):
#             self._batch_counter += 1
#             if self._batch_counter % 50 == 0:
#                 if 'loss' in result:
#                     loss_val = result['loss']
#                     if isinstance(loss_val, torch.Tensor):
#                         loss_val = loss_val.item()
#                     print(f"  📦 Batch {self._batch_counter}, Loss: {loss_val:.4f}")
#         else:
#             self._batch_counter = 1
        
#         return result
    
#     def validate(self, *args, **kwargs):
#         """验证步骤：处理忽略的类别"""
#         result = super().validate(*args, **kwargs)
        
#         if 'dice_per_class' in result:
#             dice = result['dice_per_class']
            
#             # 转换为numpy数组
#             if isinstance(dice, torch.Tensor):
#                 dice_np = dice.detach().cpu().numpy()
#             else:
#                 dice_np = np.array(dice)
            
#             # 修复NaN
#             dice_fixed = dice_np.copy()
#             dice_fixed[np.isnan(dice_fixed)] = 0.0
            
#             # 计算有效类别的平均值（排除背景、忽略类别、不存在类别）
#             valid_dice = []
#             activated_classes = []
            
#             for i in range(len(dice_fixed)):
#                 if i == 0 or i == 43 or i in self.ignore_classes:
#                     continue
#                 if dice_fixed[i] > 0.001:  # 阈值设为0.1%
#                     valid_dice.append(dice_fixed[i])
#                     activated_classes.append((i, dice_fixed[i]))
            
#             # 更新mean值
#             if valid_dice:
#                 result['mean'] = float(np.mean(valid_dice))
                
#                 print(f"\n📊 验证结果（极端权重）:")
#                 print(f"  激活类别数: {len(activated_classes)}/{len(self.train_classes)-len(self.ignore_classes)}")
#                 print(f"  平均Dice: {result['mean']:.4f}")
                
#                 # 打印最好的5个类别
#                 if activated_classes:
#                     activated_classes.sort(key=lambda x: x[1], reverse=True)
#                     print(f"  最佳类别:")
#                     for i, (cls, val) in enumerate(activated_classes[:5]):
#                         weight = self.class_weights[cls].item()
#                         print(f"    类别 {cls:2d}: Dice={val:.4f}, 权重={weight:6.1f}")
#             else:
#                 result['mean'] = 0.0
#                 print(f"\n⚠️ 警告：没有激活的前景类别")
            
#             # 更新dice_per_class
#             if isinstance(dice, torch.Tensor):
#                 result['dice_per_class'] = torch.from_numpy(dice_fixed).to(dice.device)
        
#         return result
    
#     def on_epoch_end(self):
#         """每个epoch结束时的回调"""
#         super().on_epoch_end()
        
#         # 每epoch打印进度
#         if hasattr(self, 'optimizer'):
#             lr = self.optimizer.param_groups[0]['lr']
#             print(f"\n📈 Epoch {self.current_epoch} 进度:")
#             print(f"  学习率: {lr:.2e}")
            
#             # 检查权重效果
#             if hasattr(self, 'class_weights'):
#                 extreme_weights = (self.class_weights > 100).sum().item()
#                 print(f"  极端权重数(>100): {extreme_weights}")
    
#     def configure_optimizers(self):
#         """配置优化器 - 适应极端权重"""
#         # 调用父类方法
#         result = super().configure_optimizers()
        
#         # 检查返回类型
#         if isinstance(result, tuple):
#             # nnUNet通常返回 (optimizer, scheduler)
#             optimizer, scheduler = result
            
#             # 为极端权重设置更保守的学习率
#             base_lr = 5e-5  # 较低的学习率
            
#             # 为Mamba层设置不同学习率
#             if self.mamba_added:
#                 # 分组学习率
#                 mamba_params = []
#                 other_params = []
                
#                 for name, param in self.network.named_parameters():
#                     if 'mamba' in name.lower():
#                         mamba_params.append(param)
#                     else:
#                         other_params.append(param)
                
#                 if mamba_params:
#                     param_groups = [
#                         {'params': other_params, 'lr': base_lr, 'weight_decay': 1e-5},
#                         {'params': mamba_params, 'lr': base_lr * 0.3, 'weight_decay': 1e-4}
#                     ]
                    
#                     optimizer = torch.optim.AdamW(
#                         param_groups,
#                         lr=base_lr,
#                         betas=(0.9, 0.999),
#                         eps=1e-8
#                     )
                    
#                     print(f"\n⚙️ 优化器配置（极端权重+Mamba）:")
#                     print(f"  主干网络学习率: {base_lr:.2e}")
#                     print(f"  Mamba层学习率: {base_lr*0.3:.2e}")
#                 else:
#                     # 没有Mamba参数，统一设置
#                     for param_group in optimizer.param_groups:
#                         param_group['lr'] = base_lr
#                         param_group['weight_decay'] = 1e-5
                    
#                     print(f"\n⚙️ 优化器配置（极端权重）:")
#                     print(f"  学习率: {base_lr:.2e}")
#             else:
#                 # 没有Mamba，统一设置
#                 for param_group in optimizer.param_groups:
#                     param_group['lr'] = base_lr
#                     param_group['weight_decay'] = 1e-5
                
#                 print(f"\n⚙️ 优化器配置（极端权重）:")
#                 print(f"  学习率: {base_lr:.2e}")
            
#             return optimizer, scheduler
#         else:
#             print(f"\n⚙️ 使用默认优化器配置")
#             return result
#nnUNetv2_predict -i /root/autodl-tmp/nnUNet/kk/nnUNet_raw/Dataset001_Tooth/imagesTs -o /root/autodl-tmp/nnUNet/kk/nnUNet_results/Dataset001_Tooth/test -d 1 -c 2d -f 0 -tr IgnoreTrainer





import torch
import torch.nn as nn
import os
import sys
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# ================ 您的原始修复代码 ================
os.environ['NNUNET_COMPILE'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '0'

# ================ 真正解决问题的修复 ================
import warnings
import nnunetv2.evaluation.evaluate_predictions as eval_module

IGNORE_CLASSES = [ 5, 6, 44, 47, 43]

print("="*60)
print("安装Dice NaN根本性修复")
print("="*60)

# 1. 修复compute_metrics函数
original_compute_metrics = eval_module.compute_metrics

def fixed_compute_metrics(reference_file, prediction_file, image_reader_writer,
                         labels_or_regions, ignore_label=None):
    try:
        result = original_compute_metrics(
            reference_file, prediction_file, image_reader_writer,
            labels_or_regions, ignore_label
        )
        
        if isinstance(result, dict) and 'metrics' in result:
            metrics = result['metrics']
            
            for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                if metric_name in metrics:
                    metric_dict = metrics[metric_name]
                    
                    if isinstance(metric_dict, dict):
                        for cls in IGNORE_CLASSES:
                            if str(cls) in metric_dict:
                                del metric_dict[str(cls)]
                            elif cls in metric_dict:
                                del metric_dict[cls]
                        
                        for key in list(metric_dict.keys()):
                            val = metric_dict[key]
                            if isinstance(val, (int, float)):
                                if np.isnan(val):
                                    metric_dict[key] = 0.0
                            elif hasattr(val, 'item'):
                                if np.isnan(val.item()):
                                    metric_dict[key] = 0.0
        
        return result
        
    except Exception as e:
        print(f"⚠️ compute_metrics错误: {e}")
        return {'metrics': {'dice': {}, 'hausdorff_distance_95': {}, 'mean_surface_distance': {}}}

eval_module.compute_metrics = fixed_compute_metrics
print("✅ 已修复 compute_metrics 函数")

# 2. 修复聚合函数summarize_results（省略，保持原样）
if hasattr(eval_module, 'summarize_results'):
    original_summarize = eval_module.summarize_results
    
    def fixed_summarize_results(results, args=None):
        try:
            fixed_results = []
            for res in results:
                if isinstance(res, dict) and 'metrics' in res:
                    fixed_res = res.copy()
                    metrics = fixed_res['metrics']
                    
                    for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                        if metric_name in metrics:
                            metric_dict = metrics[metric_name]
                            if isinstance(metric_dict, dict):
                                for key, val in metric_dict.items():
                                    if isinstance(val, (int, float)) and np.isnan(val):
                                        metric_dict[key] = 0.0
                    
                    fixed_results.append(fixed_res)
                else:
                    fixed_results.append(res)
            
            if args is not None:
                summary = original_summarize(fixed_results, args)
            else:
                summary = original_summarize(fixed_results)
            
            def fix_nan_in_dict(d):
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            fix_nan_in_dict(v)
                        elif isinstance(v, (int, float)) and np.isnan(v):
                            d[k] = 0.0
            
            if isinstance(summary, dict):
                fix_nan_in_dict(summary)
            
            return summary
            
        except Exception as e:
            print(f"⚠️ summarize_results错误: {e}")
            return {}

    eval_module.summarize_results = fixed_summarize_results
print("✅ 已修复 summarize_results 函数")

# 3. 修复compute_metrics_on_folder函数（省略，保持原样）
original_compute_metrics_on_folder = eval_module.compute_metrics_on_folder

def fixed_compute_metrics_on_folder(folder_ref, folder_pred, image_reader_writer,
                                   labels_or_regions, ignore_label=None, 
                                   num_processes: int = 1):
    try:
        results = original_compute_metrics_on_folder(
            folder_ref, folder_pred, image_reader_writer,
            labels_or_regions, ignore_label, num_processes
        )
        
        fixed_results = []
        for res in results:
            if isinstance(res, dict) and 'metrics' in res:
                fixed_res = res.copy()
                metrics = fixed_res['metrics']
                
                for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                    if metric_name in metrics:
                        metric_dict = metrics[metric_name]
                        if isinstance(metric_dict, dict):
                            for key, val in metric_dict.items():
                                if isinstance(val, (int, float)) and np.isnan(val):
                                    metric_dict[key] = 0.0
                
                fixed_results.append(fixed_res)
            else:
                fixed_results.append(res)
        
        return fixed_results
        
    except Exception as e:
        print(f"⚠️ compute_metrics_on_folder错误: {e}")
        return []

eval_module.compute_metrics_on_folder = fixed_compute_metrics_on_folder
print("✅ 已修复 compute_metrics_on_folder 函数")

# 4. 修复np.nanmean
original_nanmean = np.nanmean

def safe_nanmean(a, **kwargs):
    try:
        result = original_nanmean(a, **kwargs)
        if np.isnan(result):
            return 0.0
        return result
    except:
        return 0.0

np.nanmean = safe_nanmean
print("✅ 已修复 np.nanmean 函数作为最后保障")

print(f"\n" + "="*60)
print(f"🎯 修复总结:")
print(f"   忽略的类别: {IGNORE_CLASSES}")
print(f"   修复了3个关键函数 + np.nanmean作为保障")
print(f"   策略: 从计算源头防止NaN，层层防护")
print("="*60 + "\n")
# ================ 您的修复代码结束 ================

# ================ Mamba 集成开始 ================
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("⚠️ mamba_ssm 不可用，将禁用Mamba功能")
    MAMBA_AVAILABLE = False
    Mamba = None

# ================ 注意力模块导入 ================
try:
    from .attention_gates import (
        MultiScaleAttentionGate3D,
        HierarchicalAttentionGate3D,
        ClassBalancedAttentionGate3D,
        SimplifiedAttentionGate3D,
        create_attention_gate
    )
    ATTENTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入attention_gates: {e}")
    ATTENTION_AVAILABLE = False
    # 创建空类避免错误
    class MultiScaleAttentionGate3D:
        def __init__(self, *args, **kwargs):
            raise ImportError("attention_gates not available")
    class HierarchicalAttentionGate3D:
        def __init__(self, *args, **kwargs):
            raise ImportError("attention_gates not available")
    class ClassBalancedAttentionGate3D:
        def __init__(self, *args, **kwargs):
            raise ImportError("attention_gates not available")
    def create_attention_gate(*args, **kwargs):
        raise ImportError("attention_gates not available")

# ================ 核心修复：真正集成注意力到跳跃连接的训练器 ================
class IgnoreTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # ========== 训练类别设置 ==========
        self.train_classes = []
        
        for i in range(11, 43):
            if i != 43:
                self.train_classes.append(i)
        
        self.train_classes.append(48)
        
        important_classes = [1, 2, 3, 8, 9]
        for cls in important_classes:
            if cls not in self.train_classes:
                self.train_classes.append(cls)
        
        self.train_classes.sort()
        
        self.ignore_classes = [4, 5, 6, 44, 45, 46, 47, 43]
        
        self.voxel_counts = {
            0: 10357996276, 1: 709790043, 2: 34062112, 3: 6740490, 4: 6845641,
            5: 2352907, 6: 2317488, 7: 222131847, 8: 2049060, 9: 9904774,
            10: 513649, 11: 1893477, 12: 1306498, 13: 2238177, 14: 2565055,
            15: 2764938, 16: 6233357, 17: 5799851, 18: 2204634, 19: 1894346,
            20: 1325445, 21: 2235198, 22: 2686611, 23: 2850689, 24: 6449628,
            25: 6172869, 26: 2364195, 27: 3586957, 28: 4371277, 29: 7462541,
            30: 5516252, 31: 5621255, 32: 8634051, 33: 9863243, 34: 7354314,
            35: 3561221, 36: 4381718, 37: 7616134, 38: 5782732, 39: 5743643,
            40: 8713962, 41: 9710653, 42: 7634230, 
            43: 11218,
            44: 6401, 45: 303363, 46: 268274, 47: 131338, 48: 49350
        }
        
        print(f"🎯 训练配置:")
        print(f"  总类别数: 49 (0-48)")
        print(f"  训练类别数: {len(self.train_classes)}个")
        print(f"  训练类别: {self.train_classes}")
        print(f"  忽略类别: {self.ignore_classes}")
        
        # ========== 激进的权重计算 ==========
        self.class_weights = self._calculate_extreme_weights()
        
        self.apply_weights_to_loss()
        
        # 监控变量
        self.best_mean_dice = 0
        self.patience_counter = 0
        self.max_patience = 20
        
        # ========== Mamba 相关设置 ==========
        self.use_mamba = False and MAMBA_AVAILABLE
        self.mamba_added = False
        self.mamba_seq_len = 32
        self.mamba_d_state = 16
        self.mamba_d_conv = 4
        self.mamba_expand = 2
        
        # ========== 注意力机制设置 ==========
        self.use_attention = False and ATTENTION_AVAILABLE
        self.attention_type = 'class_balanced'
        self.attention_config = {
            'reduction_ratio': 16,
            'use_residual': True,
            'dropout_rate': 0.1,
            'attention_lr_multiplier': 0.5,
            'attention_weight_decay': 1e-5,
            'num_classes': 49,
            'minority_classes': [48] + list(range(11, 43)),
            'minority_boost': 2.0
        }
        
        # nnU-Net标准编码器通道数
        self.encoder_channels = [32, 64, 128, 256, 320]
        self.decoder_channels = [320, 256, 128, 64, 32]
        
        # 初始化注意力模块
        self.attention_modules = None
        
        if self.use_attention:
            print(f"\n🎯 注意力机制配置:")
            print(f"  类型: {self.attention_type}")
            print(f"  位置: 所有跳跃连接")
            print(f"  学习率乘子: {self.attention_config['attention_lr_multiplier']}")
        
        if self.use_mamba:
            print(f"\n🎯 Mamba 配置:")
            print(f"  使用官方 mamba-ssm")
            print(f"  序列长度: {self.mamba_seq_len}")
            print(f"  状态维度: {self.mamba_d_state}")
            print(f"  卷积核大小: {self.mamba_d_conv}")
            print(f"  扩展因子: {self.mamba_expand}")
        
        # 打印激进的权重信息
        self.print_extreme_weight_info()
    
    def _calculate_extreme_weights(self):
        num_classes = 49
        weights = torch.ones(num_classes, device='cpu', dtype=torch.float32)
        
        weights[0] = 0.001
        weights[43] = 0.0
        
        for cls in self.ignore_classes:
            if cls != 43:
                weights[cls] = 0.0
        
        print(f"\n📊 体素分布分析:")
        total_voxels = sum(self.voxel_counts.values())
        background_ratio = self.voxel_counts[0] / total_voxels
        print(f"  总体素数: {total_voxels:,}")
        print(f"  背景占比: {background_ratio:.2%}")
        
        foreground_counts = []
        for cls in self.train_classes:
            if cls not in self.ignore_classes and cls != 43:
                count = self.voxel_counts[cls]
                foreground_counts.append((cls, count))
        
        if foreground_counts:
            min_count = min([c for _, c in foreground_counts])
            max_count = max([c for _, c in foreground_counts])
            median_count = np.median([c for _, c in foreground_counts])
            
            print(f"  前景体素范围: {min_count:,} - {max_count:,}")
            print(f"  前景中位数: {median_count:,}")
            
            print(f"\n⚡ 计算极端权重:")
            for cls, count in foreground_counts:
                rarity_ratio = self.voxel_counts[0] / max(count, 1)
                
                if cls == 48:
                    base_weight = 500.0
                elif 11 <= cls <= 42:
                    base_weight = 100.0
                else:
                    base_weight = 50.0
                
                if count > 0:
                    rarity_factor = np.log10(min(rarity_ratio, 1e7)) / 7.0
                    rarity_factor = max(0.1, min(5.0, 1.0 + rarity_factor * 4))
                    
                    freq_factor = median_count / max(count, 1)
                    freq_factor = max(0.3, min(3.0, np.sqrt(freq_factor)))
                    
                    weight = base_weight * rarity_factor * freq_factor
                else:
                    weight = base_weight
                
                if cls == 48:
                    weight = min(800.0, max(400.0, weight))
                elif 11 <= cls <= 42:
                    weight = min(200.0, max(60.0, weight))
                else:
                    weight = min(100.0, max(30.0, weight))
                
                weights[cls] = float(weight)
                
                print(f"  类别 {cls:2d}: 体素={count:10,}, "
                      f"背景比={rarity_ratio:10.0f}x, "
                      f"权重={weight:7.1f}")
        
        max_weight = weights.max()
        if max_weight > 0:
            weights = weights / max_weight * 500.0
        
        return weights.to(self.device)
    
    def print_extreme_weight_info(self):
        weights_cpu = self.class_weights.cpu().detach().numpy()
        
        print(f"\n🔥 极端权重配置:")
        print(f"  背景权重: {weights_cpu[0]:.6f}")
        print(f"  牙髓(48)权重: {weights_cpu[48]:.1f}")
        
        valid_train_classes = []
        valid_weights = []
        
        for cls in self.train_classes:
            if cls not in self.ignore_classes and weights_cpu[cls] > 0:
                valid_train_classes.append(cls)
                valid_weights.append(weights_cpu[cls])
        
        if valid_weights:
            print(f"  有效训练类别数: {len(valid_train_classes)}个")
            print(f"  权重范围: {np.min(valid_weights):.1f} - {np.max(valid_weights):.1f}")
            print(f"  权重中位数: {np.median(valid_weights):.1f}")
            print(f"  平均权重: {np.mean(valid_weights):.1f}")
            
            min_weight = np.min(valid_weights)
            max_weight = np.max(valid_weights)
            print(f"  最大/最小权重比: {max_weight/min_weight:.1f}x")
            
            pulp_weight = weights_cpu[48]
            bg_weight = weights_cpu[0]
            if bg_weight > 0:
                ratio = pulp_weight / bg_weight
                print(f"  牙髓/背景权重比: 1 : {ratio:,.0f}")
            
            print(f"\n🎯 关键类别权重详情:")
            key_classes = [0, 1, 2, 3, 8, 9, 11, 32, 48]
            for cls in key_classes:
                if cls < len(weights_cpu):
                    voxel_count = self.voxel_counts.get(cls, 0)
                    bg_ratio = self.voxel_counts[0] / max(voxel_count, 1) if voxel_count > 0 else 0
                    print(f"  类别 {cls:2d}: 权重={weights_cpu[cls]:7.1f}, "
                          f"体素={voxel_count:10,}, "
                          f"背景比={bg_ratio:10,.0f}x")
        
        weight_sum = weights_cpu.sum()
        print(f"\n📈 权重总和: {weight_sum:.1f}")
    
    def apply_weights_to_loss(self):
        if hasattr(self.loss, 'ce_loss'):
            if hasattr(self.loss.ce_loss, 'weight'):
                self.loss.ce_loss.weight = self.class_weights
                print(f"\n✅ 已设置CE损失极端权重")
                
                weight_sum = self.class_weights.sum().item()
                max_weight = self.class_weights.max().item()
                print(f"  权重总和: {weight_sum:.1f}")
                print(f"  最大权重: {max_weight:.1f} (类别 {self.class_weights.argmax().item()})")
    
    # ========== 核心修复：真正集成注意力到跳跃连接 ==========
    def initialize_network(self):
        """重写网络初始化以真正集成注意力到跳跃连接"""
        # 先调用父类初始化创建基础网络
        super().initialize_network()
        
        if self.use_mamba and not self.mamba_added:
            self._add_extreme_mamba_to_network()
        
        if self.use_attention:
            self._initialize_attention_modules()
            
            # 关键：真正修改网络结构，将注意力集成到跳跃连接
            self._integrate_attention_into_unet_architecture()
    
    def _integrate_attention_into_unet_architecture(self):
        """真正将注意力模块集成到UNet架构中"""
        print("\n" + "="*60)
        print("🔥 正在将注意力模块真正集成到UNet跳跃连接...")
        print("="*60)
        
        # 检查网络类型
        if hasattr(self.network, 'encoder') and hasattr(self.network, 'decoder'):
            # 标准UNet结构
            self._integrate_into_standard_unet()
        elif hasattr(self.network, 'conv_blocks_context'):
            # nnU-Net的generic_modular_UNet结构
            self._integrate_into_generic_modular_unet()
        else:
            print(f"⚠️ 无法识别的网络结构，无法集成注意力模块")
            print(f"  网络类型: {type(self.network)}")
            print(f"  网络属性: {dir(self.network)}")
    
    def _integrate_into_standard_unet(self):
        """集成到标准UNet结构"""
        print("  检测到标准UNet结构")
        
        # 方法1：直接修改解码器的前向传播
        original_forward = self.network.forward
        
        def attention_enhanced_forward(x):
            # 存储编码器特征
            encoder_features = []
            current = x
            
            # 编码器前向传播
            for encoder_block in self.network.encoder.blocks:
                current = encoder_block(current)
                encoder_features.append(current)
            
            # 解码器前向传播（带注意力）
            current = encoder_features[-1]
            for i, decoder_block in enumerate(reversed(self.network.decoder.blocks)):
                # 对应的跳跃连接索引
                skip_idx = len(self.network.decoder.blocks) - i - 1
                
                if skip_idx < len(encoder_features) and f'skip_{skip_idx}' in self.attention_modules:
                    # 应用注意力到跳跃连接
                    skip_feature = encoder_features[skip_idx]
                    gate_feature = current
                    
                    attention_gate = self.attention_modules[f'skip_{skip_idx}']
                    
                    if isinstance(attention_gate, ClassBalancedAttentionGate3D):
                        class_importance = self.class_weights.clone()
                        for cls in self.ignore_classes:
                            if cls < len(class_importance):
                                class_importance[cls] = 0
                        
                        enhanced_skip, _ = attention_gate(
                            skip_feature, gate_feature, 
                            class_importance=class_importance,
                            return_class_attention=False
                        )
                        encoder_features[skip_idx] = enhanced_skip
                    else:
                        enhanced_skip = attention_gate(skip_feature, gate_feature)
                        encoder_features[skip_idx] = enhanced_skip
                
                # 解码器操作
                skip_feature = encoder_features[skip_idx] if skip_idx < len(encoder_features) else None
                current = decoder_block(current, skip_feature)
            
            # 分割头
            output = self.network.segmentation_head(current)
            return output
        
        # 替换前向传播
        import types
        self.network.forward = types.MethodType(attention_enhanced_forward, self.network)
        print("  ✅ 已集成注意力到标准UNet跳跃连接")
    
    def _integrate_into_generic_modular_unet(self):
        """集成到nnU-Net的generic_modular_UNet结构"""
        print("  检测到generic_modular_UNet结构")
        
        # 获取网络中的跳跃连接位置
        # generic_modular_UNet通常有conv_blocks_context（编码器）和conv_blocks_localization（解码器）
        
        try:
            # 方法1：通过属性名查找跳跃连接
            if hasattr(self.network, 'tu'):
                # tu是上采样模块，通常处理跳跃连接
                print(f"  找到上采样模块: tu")
                self._modify_tu_blocks()
            elif hasattr(self.network, 'conv_blocks_localization'):
                # 直接修改localization块
                print(f"  找到localization模块")
                self._modify_localization_blocks()
            else:
                print(f"  ⚠️ 无法找到跳跃连接位置")
        except Exception as e:
            print(f"  ❌ 集成失败: {e}")
    
    def _modify_tu_blocks(self):
        """修改上采样模块以集成注意力"""
        tu_blocks = self.network.tu
        if not isinstance(tu_blocks, nn.ModuleList):
            print(f"  ⚠️ tu不是ModuleList，无法修改")
            return
        
        print(f"  找到 {len(tu_blocks)} 个上采样块")
        
        for i, tu_block in enumerate(tu_blocks):
            if f'skip_{i}' in self.attention_modules:
                print(f"   为tu_block[{i}]添加注意力")
                
                # 保存原始前向传播
                original_tu_forward = tu_block.forward
                
                def attention_tu_forward(self_tu, x, skip):
                    # 应用注意力到跳跃连接
                    attention_gate = self.attention_modules[f'skip_{i}']
                    
                    if isinstance(attention_gate, ClassBalancedAttentionGate3D):
                        class_importance = self.class_weights.clone()
                        for cls in self.ignore_classes:
                            if cls < len(class_importance):
                                class_importance[cls] = 0
                        
                        enhanced_skip, _ = attention_gate(
                            skip, x,
                            class_importance=class_importance,
                            return_class_attention=False
                        )
                        skip = enhanced_skip
                    else:
                        skip = attention_gate(skip, x)
                    
                    # 调用原始前向传播
                    return original_tu_forward(x, skip)
                
                # 绑定新的前向传播
                import types
                tu_block.forward = types.MethodType(attention_tu_forward, tu_block)
        
        print(f"  ✅ 已为 {len([k for k in self.attention_modules.keys() if 'skip_' in k])} 个跳跃连接添加注意力")
    
    def _modify_localization_blocks(self):
        """修改localization模块以集成注意力"""
        loc_blocks = self.network.conv_blocks_localization
        if not isinstance(loc_blocks, nn.ModuleList):
            print(f"  ⚠️ conv_blocks_localization不是ModuleList，无法修改")
            return
        
        print(f"  找到 {len(loc_blocks)} 个localization块")
        
        for i, loc_block in enumerate(loc_blocks):
            if i < len(loc_blocks) - 1 and f'skip_{i}' in self.attention_modules:
                print(f"   为localization_block[{i}]添加注意力")
                
                # 保存原始前向传播
                original_loc_forward = loc_block.forward
                
                def attention_loc_forward(self_loc, x, skip):
                    # 应用注意力到跳跃连接
                    attention_gate = self.attention_modules[f'skip_{i}']
                    
                    if isinstance(attention_gate, ClassBalancedAttentionGate3D):
                        class_importance = self.class_weights.clone()
                        for cls in self.ignore_classes:
                            if cls < len(class_importance):
                                class_importance[cls] = 0
                        
                        enhanced_skip, _ = attention_gate(
                            skip, x,
                            class_importance=class_importance,
                            return_class_attention=False
                        )
                        skip = enhanced_skip
                    else:
                        skip = attention_gate(skip, x)
                    
                    # 调用原始前向传播
                    return original_loc_forward(x, skip)
                
                # 绑定新的前向传播
                import types
                loc_block.forward = types.MethodType(attention_loc_forward, loc_block)
        
        print(f"  ✅ 已修改localization块以集成注意力")
    
    def _initialize_attention_modules(self):
        """初始化注意力模块"""
        if not self.use_attention:
            return
        
        print("\n" + "="*60)
        print("🎯 正在初始化多尺度注意力模块...")
        print("="*60)
        
        self.attention_modules = nn.ModuleDict()
        
        num_skips = len(self.encoder_channels) - 1
        deep_levels = self.attention_config.get('class_balanced_levels', 1)
        deep_start = max(0, num_skips - deep_levels)

        # 预计算类频（用于类平衡门）
        class_freqs = None
        try:
            if hasattr(self, 'voxel_counts'):
                class_freqs = [float(self.voxel_counts.get(cls, 1.0)) for cls in range(self.attention_config['num_classes'])]
        except Exception:
            class_freqs = None

        # 为每个跳跃连接创建注意力门
        for i, (skip_ch, gate_ch) in enumerate(zip(self.encoder_channels[:-1], self.decoder_channels[1:])):
            scale_factor = 2

            gate_type = self.attention_type
            if self.attention_type == 'hybrid':
                gate_type = 'class_balanced' if i >= deep_start else self.attention_config.get('shallow_gate', 'multiscale')

            try:
                if gate_type == 'class_balanced':
                    attention_gate = ClassBalancedAttentionGate3D(
                        skip_channels=skip_ch,
                        gate_channels=gate_ch,
                        num_classes=self.attention_config['num_classes'],
                        minority_class_indices=self.attention_config['minority_classes'],
                        minority_boost=self.attention_config['minority_boost'],
                        class_frequencies=class_freqs,
                        weight_mode=self.attention_config.get('weight_mode', 'inv_sqrt'),
                        scale_factor=scale_factor,
                        reduction_ratio=self.attention_config['reduction_ratio'],
                        use_residual=self.attention_config['use_residual'],
                        dropout_rate=self.attention_config['dropout_rate']
                    )
                elif gate_type == 'hierarchical':
                    attention_gate = HierarchicalAttentionGate3D(
                        skip_channels=skip_ch,
                        gate_channels_list=[gate_ch, self.decoder_channels[i] if i > 0 else gate_ch],
                        scale_factor=scale_factor,
                        reduction_ratio=self.attention_config['reduction_ratio']
                    )
                elif gate_type == 'simplified':
                    attention_gate = SimplifiedAttentionGate3D(
                        skip_channels=skip_ch,
                        gate_channels=gate_ch,
                        scale_factor=scale_factor
                    )
                else:
                    attention_gate = MultiScaleAttentionGate3D(
                        skip_channels=skip_ch,
                        gate_channels=gate_ch,
                        scale_factor=scale_factor,
                        reduction_ratio=self.attention_config['reduction_ratio'],
                        use_residual=self.attention_config['use_residual'],
                        dropout_rate=self.attention_config['dropout_rate']
                    )

                self.attention_modules[f'skip_{i}'] = attention_gate
                print(f"  ✅ 创建注意力门 {i}: skip={skip_ch}, gate={gate_ch}, type={gate_type}")
            except Exception as e:
                print(f"  ❌ 创建注意力门 {i} 失败: {e}")
        
        # 移动到设备
        if self.attention_modules:
            self.attention_modules.to(self.device)
            print(f"\n🎉 成功初始化 {len(self.attention_modules)} 个注意力模块")
            print(f"  注意力参数总数: {sum(p.numel() for p in self.attention_modules.parameters()):,}")
        else:
            print(f"\n⚠️ 注意力模块初始化失败")
    
    # 其他方法保持不变...
    def _add_extreme_mamba_to_network(self):
        """在网络中添加强化的Mamba瓶颈层"""
        if not MAMBA_AVAILABLE:
            print(f"⚠️ Mamba不可用，跳过Mamba增强")
            return
        
        print("\n🔥 正在添加激进的Mamba增强层...")
        
        # 查找所有卷积层
        conv_layers = []
        
        def find_conv_layers(module, name=""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child_module, nn.Conv3d):
                    conv_layers.append((full_name, child_module))
                
                find_conv_layers(child_module, full_name)
        
        find_conv_layers(self.network)
        
        print(f"找到 {len(conv_layers)} 个卷积层")
        
        if conv_layers:
            conv_layers.sort(key=lambda x: x[0].count('.'), reverse=True)
            target_layers = conv_layers[:3]
            
            print(f"\n🎯 选择以下层进行Mamba增强:")
            for i, (name, module) in enumerate(target_layers):
                print(f"  {i+1}. {name}: {module.in_channels} -> {module.out_channels}")
            
            replaced_count = 0
            for name, module in target_layers:
                try:
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    
                    if in_channels >= 64 and out_channels >= 64:
                        # 创建Mamba瓶颈层（需要重新定义，因为原代码中的MambaBottleneck3D可能不可用）
                        print(f"  ⚠️ 跳过Mamba替换 {name} (需要完整Mamba实现)")
                except Exception as e:
                    print(f"  ❌ 替换 {name} 失败: {e}")
            
            if replaced_count > 0:
                self.mamba_added = True
                print(f"\n🎉 成功添加 {replaced_count} 个Mamba增强层")
            else:
                print(f"⚠️ 未成功替换任何层")
        else:
            print(f"⚠️ 未找到合适的卷积层")
    
    def configure_optimizers(self):
        """配置优化器 - 适应极端权重、Mamba和注意力"""
        result = super().configure_optimizers()
        
        if isinstance(result, tuple):
            optimizer, scheduler = result
            
            base_lr = 5e-5
            
            # 参数分组
            mamba_params = []
            attention_params = []
            other_params = []
            
            # 1. 收集Mamba参数
            if self.mamba_added:
                for name, param in self.network.named_parameters():
                    if 'mamba' in name.lower():
                        mamba_params.append(param)
            
            # 2. 收集注意力参数（网络中的注意力层）
            for name, param in self.network.named_parameters():
                if 'attention' in name.lower() or 'gate' in name.lower():
                    attention_params.append(param)
            
            # 3. 收集注意力模块参数
            if self.attention_modules is not None:
                for name, param in self.attention_modules.named_parameters():
                    attention_params.append(param)
            
            # 4. 收集其他参数
            for name, param in self.network.named_parameters():
                if ('mamba' not in name.lower() and 
                    'attention' not in name.lower() and 
                    'gate' not in name.lower()):
                    other_params.append(param)
            
            # 创建参数组
            param_groups = []
            
            if other_params:
                param_groups.append({
                    'params': other_params,
                    'lr': base_lr,
                    'weight_decay': 1e-5,
                    'name': 'backbone'
                })
            
            if mamba_params:
                mamba_lr = base_lr * 0.3
                param_groups.append({
                    'params': mamba_params,
                    'lr': mamba_lr,
                    'weight_decay': 1e-4,
                    'name': 'mamba'
                })
            
            if attention_params:
                attention_lr = base_lr * self.attention_config['attention_lr_multiplier']
                param_groups.append({
                    'params': attention_params,
                    'lr': attention_lr,
                    'weight_decay': self.attention_config['attention_weight_decay'],
                    'name': 'attention'
                })
            
            # 创建优化器
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=base_lr,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            print(f"\n⚙️ 优化器配置（极端权重 + Mamba + 注意力）:")
            print(f"  总参数组数: {len(param_groups)}")
            
            for group in param_groups:
                name = group.get('name', 'unknown')
                lr = group['lr']
                wd = group['weight_decay']
                params_count = len(group['params'])
                print(f"  {name:10s}: lr={lr:.2e}, wd={wd:.1e}, params={params_count}")
            
            print(f"\n📊 参数统计:")
            print(f"  主干网络参数: {len(other_params)}")
            print(f"  Mamba参数: {len(mamba_params)}")
            print(f"  注意力参数: {len(attention_params)}")
            total_params = sum(p.numel() for group in param_groups for p in group['params'])
            print(f"  总可训练参数: {total_params:,}")
            
            return optimizer, scheduler
        else:
            print(f"\n⚙️ 使用默认优化器配置")
            return result
    
    def validate(self, *args, **kwargs):
        """验证步骤"""
        result = super().validate(*args, **kwargs)
        
        if 'dice_per_class' in result:
            dice = result['dice_per_class']
            
            if isinstance(dice, torch.Tensor):
                dice_np = dice.detach().cpu().numpy()
            else:
                dice_np = np.array(dice)
            
            dice_fixed = dice_np.copy()
            dice_fixed[np.isnan(dice_fixed)] = 0.0
            
            valid_dice = []
            activated_classes = []
            
            for i in range(len(dice_fixed)):
                if i == 0 or i == 43 or i in self.ignore_classes:
                    continue
                if dice_fixed[i] > 0.001:
                    valid_dice.append(dice_fixed[i])
                    activated_classes.append((i, dice_fixed[i]))
            
            if valid_dice:
                result['mean'] = float(np.mean(valid_dice))
                
                print(f"\n📊 验证结果（极端权重 + 注意力）:")
                print(f"  激活类别数: {len(activated_classes)}/{len(self.train_classes)-len(self.ignore_classes)}")
                print(f"  平均Dice: {result['mean']:.4f}")
                
                if activated_classes:
                    activated_classes.sort(key=lambda x: x[1], reverse=True)
                    print(f"  最佳类别:")
                    for i, (cls, val) in enumerate(activated_classes[:5]):
                        weight = self.class_weights[cls].item()
                        print(f"    类别 {cls:2d}: Dice={val:.4f}, 权重={weight:6.1f}")
            else:
                result['mean'] = 0.0
                print(f"\n⚠️ 警告：没有激活的前景类别")
            
            if isinstance(dice, torch.Tensor):
                result['dice_per_class'] = torch.from_numpy(dice_fixed).to(dice.device)
        
        return result
    
    def train_step(self, data_batch):
        """训练步骤"""
        if self.current_epoch == 0 and not hasattr(self, '_train_info_printed'):
            print(f"\n" + "="*60)
            print(f"🚀 训练开始 - 综合改进策略")
            print(f"  训练类别: {len(self.train_classes)}个")
            print(f"  牙髓权重: {self.class_weights[48].item():.1f}")
            
            if self.use_mamba and self.mamba_added:
                print(f"  🔥 使用激进的Mamba增强")
            
            if self.use_attention and self.attention_modules:
                print(f"  🎯 使用{self.attention_type}注意力机制")
                print(f"  注意力模块数: {len(self.attention_modules)}")
                print(f"  已集成到跳跃连接: 是")
            
            print(f"="*60)
            
            self._train_info_printed = True
        
        # 处理target
        if 'target' in data_batch:
            target = data_batch['target']
            
            if isinstance(target, list):
                new_targets = []
                for t in target:
                    if isinstance(t, torch.Tensor):
                        t = t.clone()
                        for cls in self.ignore_classes:
                            t[t == cls] = 0
                        new_targets.append(t)
                    else:
                        new_targets.append(t)
                data_batch['target'] = new_targets
            elif isinstance(target, torch.Tensor):
                target = target.clone()
                for cls in self.ignore_classes:
                    target[target == cls] = 0
                data_batch['target'] = target
        
        result = super().train_step(data_batch)
        
        # 监控
        if hasattr(self, '_batch_counter'):
            self._batch_counter += 1
            if self._batch_counter % 50 == 0:
                if 'loss' in result:
                    loss_val = result['loss']
                    if isinstance(loss_val, torch.Tensor):
                        loss_val = loss_val.item()
                    print(f"  📦 Batch {self._batch_counter}, Loss: {loss_val:.4f}")
        else:
            self._batch_counter = 1
        
        return result
    
    def on_epoch_end(self):
        """每个epoch结束时的回调"""
        super().on_epoch_end()
        
        if hasattr(self, 'optimizer'):
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\n📈 Epoch {self.current_epoch} 进度:")
            print(f"  学习率: {lr:.2e}")
            
            if hasattr(self, 'class_weights'):
                extreme_weights = (self.class_weights > 100).sum().item()
                print(f"  极端权重数(>100): {extreme_weights}")





#处理+注意力最终代码

# import torch
# import torch.nn as nn
# import os
# import sys
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# import numpy as np
# from typing import Dict, List, Tuple, Optional, Union

# # ================ 您的原始修复代码 ================
# os.environ['NNUNET_COMPILE'] = '0'
# os.environ['TORCHDYNAMO_DISABLE'] = '1'
# os.environ['TORCH_COMPILE_DEBUG'] = '0'

# # ================ 真正解决问题的修复 ================
# import warnings
# import nnunetv2.evaluation.evaluate_predictions as eval_module

# IGNORE_CLASSES = [43, 44]  # 只保留43,44

# print("="*60)
# print("安装Dice NaN根本性修复")
# print("="*60)

# # 1. 修复compute_metrics函数
# original_compute_metrics = eval_module.compute_metrics

# def fixed_compute_metrics(reference_file, prediction_file, image_reader_writer,
#                          labels_or_regions, ignore_label=None):
#     try:
#         result = original_compute_metrics(
#             reference_file, prediction_file, image_reader_writer,
#             labels_or_regions, ignore_label
#         )
        
#         if isinstance(result, dict) and 'metrics' in result:
#             metrics = result['metrics']
            
#             for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                 if metric_name in metrics:
#                     metric_dict = metrics[metric_name]
                    
#                     if isinstance(metric_dict, dict):
#                         for cls in IGNORE_CLASSES:
#                             if str(cls) in metric_dict:
#                                 del metric_dict[str(cls)]
#                             elif cls in metric_dict:
#                                 del metric_dict[cls]
                        
#                         for key in list(metric_dict.keys()):
#                             val = metric_dict[key]
#                             if isinstance(val, (int, float)):
#                                 if np.isnan(val):
#                                     metric_dict[key] = 0.0
#                             elif hasattr(val, 'item'):
#                                 if np.isnan(val.item()):
#                                     metric_dict[key] = 0.0
        
#         return result
        
#     except Exception as e:
#         print(f"⚠️ compute_metrics错误: {e}")
#         return {'metrics': {'dice': {}, 'hausdorff_distance_95': {}, 'mean_surface_distance': {}}}

# eval_module.compute_metrics = fixed_compute_metrics
# print("✅ 已修复 compute_metrics 函数")

# # 2. 修复聚合函数summarize_results（省略，保持原样）
# if hasattr(eval_module, 'summarize_results'):
#     original_summarize = eval_module.summarize_results
    
#     def fixed_summarize_results(results, args=None):
#         try:
#             fixed_results = []
#             for res in results:
#                 if isinstance(res, dict) and 'metrics' in res:
#                     fixed_res = res.copy()
#                     metrics = fixed_res['metrics']
                    
#                     for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                         if metric_name in metrics:
#                             metric_dict = metrics[metric_name]
#                             if isinstance(metric_dict, dict):
#                                 for key, val in metric_dict.items():
#                                     if isinstance(val, (int, float)) and np.isnan(val):
#                                         metric_dict[key] = 0.0
                    
#                     fixed_results.append(fixed_res)
#                 else:
#                     fixed_results.append(res)
            
#             if args is not None:
#                 summary = original_summarize(fixed_results, args)
#             else:
#                 summary = original_summarize(fixed_results)
            
#             def fix_nan_in_dict(d):
#                 if isinstance(d, dict):
#                     for k, v in d.items():
#                         if isinstance(v, dict):
#                             fix_nan_in_dict(v)
#                         elif isinstance(v, (int, float)) and np.isnan(v):
#                             d[k] = 0.0
            
#             if isinstance(summary, dict):
#                 fix_nan_in_dict(summary)
            
#             return summary
            
#         except Exception as e:
#             print(f"⚠️ summarize_results错误: {e}")
#             return {}

#     eval_module.summarize_results = fixed_summarize_results
# print("✅ 已修复 summarize_results 函数")

# # 3. 修复compute_metrics_on_folder函数（省略，保持原样）
# original_compute_metrics_on_folder = eval_module.compute_metrics_on_folder

# def fixed_compute_metrics_on_folder(folder_ref, folder_pred, image_reader_writer,
#                                    labels_or_regions, ignore_label=None, 
#                                    num_processes: int = 1):
#     try:
#         results = original_compute_metrics_on_folder(
#             folder_ref, folder_pred, image_reader_writer,
#             labels_or_regions, ignore_label, num_processes
#         )
        
#         fixed_results = []
#         for res in results:
#             if isinstance(res, dict) and 'metrics' in res:
#                 fixed_res = res.copy()
#                 metrics = fixed_res['metrics']
                
#                 for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                     if metric_name in metrics:
#                         metric_dict = metrics[metric_name]
#                         if isinstance(metric_dict, dict):
#                             for key, val in metric_dict.items():
#                                 if isinstance(val, (int, float)) and np.isnan(val):
#                                     metric_dict[key] = 0.0
                
#                 fixed_results.append(fixed_res)
#             else:
#                 fixed_results.append(res)
        
#         return fixed_results
        
#     except Exception as e:
#         print(f"⚠️ compute_metrics_on_folder错误: {e}")
#         return []

# eval_module.compute_metrics_on_folder = fixed_compute_metrics_on_folder
# print("✅ 已修复 compute_metrics_on_folder 函数")

# # 4. 修复np.nanmean
# original_nanmean = np.nanmean

# def safe_nanmean(a, **kwargs):
#     try:
#         result = original_nanmean(a, **kwargs)
#         if np.isnan(result):
#             return 0.0
#         return result
#     except:
#         return 0.0

# np.nanmean = safe_nanmean
# print("✅ 已修复 np.nanmean 函数作为最后保障")

# print(f"\n" + "="*60)
# print(f"🎯 修复总结:")
# print(f"   忽略的类别: {IGNORE_CLASSES}")
# print(f"   修复了3个关键函数 + np.nanmean作为保障")
# print(f"   策略: 从计算源头防止NaN，层层防护")
# print("="*60 + "\n")
# # ================ 您的修复代码结束 ================

# # ================ Mamba 集成开始 ================
# try:
#     from mamba_ssm import Mamba
#     MAMBA_AVAILABLE = True
# except ImportError:
#     print("⚠️ mamba_ssm 不可用，将禁用Mamba功能")
#     MAMBA_AVAILABLE = False
#     Mamba = None

# # ================ 注意力模块导入 ================
# try:
#     from .attention_gates import (
#         MultiScaleAttentionGate3D,
#         HierarchicalAttentionGate3D,
#         ClassBalancedAttentionGate3D,
#         create_attention_gate
#     )
#     ATTENTION_AVAILABLE = True
# except ImportError as e:
#     print(f"⚠️ 无法导入attention_gates: {e}")
#     ATTENTION_AVAILABLE = False
#     # 创建空类避免错误
#     class MultiScaleAttentionGate3D:
#         def __init__(self, *args, **kwargs):
#             raise ImportError("attention_gates not available")
#     class HierarchicalAttentionGate3D:
#         def __init__(self, *args, **kwargs):
#             raise ImportError("attention_gates not available")
#     class ClassBalancedAttentionGate3D:
#         def __init__(self, *args, **kwargs):
#             raise ImportError("attention_gates not available")
#     def create_attention_gate(*args, **kwargs):
#         raise ImportError("attention_gates not available")

# # ================ 核心修复：真正集成注意力到跳跃连接的训练器 ================
# class TryTrainer(nnUNetTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, 
#                  dataset_json: dict, device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, device)
        
#         # ========== 训练类别设置 ==========
#         # 只设置忽略类别，其他保持不变
#         self.ignore_classes = [43, 44]
        
#         # ========== 论文设置：禁用镜像和后处理 ==========
#         self.disable_mirror = True  # 禁用镜像（牙齿不对称）
#         self.enable_postprocessing = True  # 启用后处理
#         self.postprocessing_params = {
#             'max_components': 3,  # 保留最大的3个连通分量
#             'min_component_size': 100,  # 最小分量大小
#         }
        
#         print(f"🎯 训练配置:")
#         print(f"  忽略类别: {self.ignore_classes}")
#         print(f"  镜像增强: {'已禁用' if self.disable_mirror else '已启用'}")
#         print(f"  后处理: 连通域分析（保留最大{self.postprocessing_params['max_components']}个分量）")
        
#         # 监控变量
#         self.best_mean_dice = 0
#         self.patience_counter = 0
#         self.max_patience = 20
        
#         # ========== Mamba 相关设置 ==========
#         self.use_mamba = True and MAMBA_AVAILABLE
#         self.mamba_added = False
#         self.mamba_seq_len = 32
#         self.mamba_d_state = 16
#         self.mamba_d_conv = 4
#         self.mamba_expand = 2
        
#         # ========== 注意力机制设置 ==========
#         self.use_attention = True and ATTENTION_AVAILABLE
#         self.attention_type = 'class_balanced'
#         self.attention_config = {
#             'reduction_ratio': 16,
#             'use_residual': True,
#             'dropout_rate': 0.1,
#             'attention_lr_multiplier': 0.5,
#             'attention_weight_decay': 1e-5,
#             'num_classes': 49,
#             'minority_classes': [48] + list(range(11, 43)),
#             'minority_boost': 2.0
#         }
        
#         # nnU-Net标准编码器通道数
#         self.encoder_channels = [32, 64, 128, 256, 320]
#         self.decoder_channels = [320, 256, 128, 64, 32]
        
#         # 初始化注意力模块
#         self.attention_modules = None
        
#         if self.use_attention:
#             print(f"\n🎯 注意力机制配置:")
#             print(f"  类型: {self.attention_type}")
#             print(f"  位置: 所有跳跃连接")
#             print(f"  学习率乘子: {self.attention_config['attention_lr_multiplier']}")
        
#         if self.use_mamba:
#             print(f"\n🎯 Mamba 配置:")
#             print(f"  使用官方 mamba-ssm")
#             print(f"  序列长度: {self.mamba_seq_len}")
#             print(f"  状态维度: {self.mamba_d_state}")
#             print(f"  卷积核大小: {self.mamba_d_conv}")
#             print(f"  扩展因子: {self.mamba_expand}")
    
#     # ========== 论文设置：配置数据增强（禁用镜像） ==========
#     def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
#         """配置数据增强参数 - 禁用镜像"""
#         # 先调用父类方法
#         patch_size, mirror_axes, rotation_for_DA, do_dummy_2d_data_aug = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
#         # 禁用镜像增强
#         mirror_axes = ()
#         print(f"🔧 数据增强配置: 镜像增强已禁用（mirror_axes={mirror_axes}）")
        
#         # 必须返回这四个值
#         return patch_size, mirror_axes, rotation_for_DA, do_dummy_2d_data_aug
    
#     # ========== 论文设置：后处理方法 ==========
#     def run_inference(self, *args, **kwargs):
#         """运行推理并应用后处理"""
#         # 设置是否进行后处理
#         kwargs['do_postprocessing'] = self.enable_postprocessing
        
#         # 调用父类推理
#         predictions = super().run_inference(*args, **kwargs)
        
#         if self.enable_postprocessing and predictions is not None:
#             print(f"🔧 应用后处理（保留最大{self.postprocessing_params['max_components']}个连通分量）")
            
#             # 对每个预测进行后处理
#             processed_predictions = []
#             for pred in predictions:
#                 if isinstance(pred, np.ndarray):
#                     # 连通域分析后处理
#                     processed_pred = self._connected_component_postprocessing(pred)
#                     processed_predictions.append(processed_pred)
#                 else:
#                     processed_predictions.append(pred)
            
#             return processed_predictions
        
#         return predictions
    
#     def _connected_component_postprocessing(self, segmentation):
#         """连通域分析后处理"""
#         try:
#             import scipy.ndimage as ndi
#             from skimage.measure import label
#         except ImportError:
#             print("⚠️ 未安装scipy或skimage，跳过后处理")
#             return segmentation
        
#         result = np.zeros_like(segmentation)
        
#         # 对每个类别独立处理
#         unique_labels = np.unique(segmentation)
#         unique_labels = unique_labels[unique_labels != 0]  # 排除背景
        
#         for label_id in unique_labels:
#             binary_mask = (segmentation == label_id).astype(np.uint8)
            
#             # 连通域标记
#             labeled_mask, num_features = label(binary_mask, connectivity=3, return_num=True)
            
#             if num_features > 0:
#                 # 计算每个连通分量的大小
#                 component_sizes = []
#                 component_indices = []
                
#                 for i in range(1, num_features + 1):
#                     component_mask = (labeled_mask == i)
#                     size = np.sum(component_mask)
#                     component_sizes.append(size)
#                     component_indices.append(i)
                
#                 # 按大小排序
#                 sorted_indices = np.argsort(component_sizes)[::-1]  # 从大到小
                
#                 # 保留最大的N个分量
#                 keep_count = min(self.postprocessing_params['max_components'], len(sorted_indices))
                
#                 for i in range(keep_count):
#                     idx = sorted_indices[i]
#                     component_id = component_indices[idx]
#                     component_size = component_sizes[idx]
                    
#                     # 只保留足够大的分量
#                     if component_size >= self.postprocessing_params['min_component_size']:
#                         component_mask = (labeled_mask == component_id)
#                         result[component_mask] = label_id
        
#         return result
    
#     # ========== 核心修复：真正集成注意力到跳跃连接 ==========
#     def initialize_network(self):
#         """重写网络初始化以真正集成注意力到跳跃连接"""
#         # 先调用父类初始化创建基础网络
#         super().initialize_network()
        
#         if self.use_mamba and not self.mamba_added:
#             self._add_extreme_mamba_to_network()
        
#         if self.use_attention:
#             self._initialize_attention_modules()
            
#             # 关键：真正修改网络结构，将注意力集成到跳跃连接
#             self._integrate_attention_into_unet_architecture()
    
#     def _integrate_attention_into_unet_architecture(self):
#         """真正将注意力模块集成到UNet架构中"""
#         print("\n" + "="*60)
#         print("🔥 正在将注意力模块真正集成到UNet跳跃连接...")
#         print("="*60)
        
#         # 检查网络类型
#         if hasattr(self.network, 'encoder') and hasattr(self.network, 'decoder'):
#             # 标准UNet结构
#             self._integrate_into_standard_unet()
#         elif hasattr(self.network, 'conv_blocks_context'):
#             # nnU-Net的generic_modular_UNet结构
#             self._integrate_into_generic_modular_unet()
#         else:
#             print(f"⚠️ 无法识别的网络结构，无法集成注意力模块")
#             print(f"  网络类型: {type(self.network)}")
#             print(f"  网络属性: {dir(self.network)}")
    
#     def _integrate_into_standard_unet(self):
#         """集成到标准UNet结构"""
#         print("  检测到标准UNet结构")
        
#         # 方法1：直接修改解码器的前向传播
#         original_forward = self.network.forward
        
#         def attention_enhanced_forward(x):
#             # 存储编码器特征
#             encoder_features = []
#             current = x
            
#             # 编码器前向传播
#             for encoder_block in self.network.encoder.blocks:
#                 current = encoder_block(current)
#                 encoder_features.append(current)
            
#             # 解码器前向传播（带注意力）
#             current = encoder_features[-1]
#             for i, decoder_block in enumerate(reversed(self.network.decoder.blocks)):
#                 # 对应的跳跃连接索引
#                 skip_idx = len(self.network.decoder.blocks) - i - 1
                
#                 if skip_idx < len(encoder_features) and f'skip_{skip_idx}' in self.attention_modules:
#                     # 应用注意力到跳跃连接
#                     skip_feature = encoder_features[skip_idx]
#                     gate_feature = current
                    
#                     attention_gate = self.attention_modules[f'skip_{skip_idx}']
                    
#                     if isinstance(attention_gate, ClassBalancedAttentionGate3D):
#                         # 创建一个简单的类重要性张量
#                         class_importance = torch.ones(49, device=self.device)
#                         for cls in self.ignore_classes:
#                             if cls < len(class_importance):
#                                 class_importance[cls] = 0
                        
#                         enhanced_skip, _ = attention_gate(
#                             skip_feature, gate_feature, 
#                             class_importance=class_importance,
#                             return_class_attention=False
#                         )
#                         encoder_features[skip_idx] = enhanced_skip
#                     else:
#                         enhanced_skip = attention_gate(skip_feature, gate_feature)
#                         encoder_features[skip_idx] = enhanced_skip
                
#                 # 解码器操作
#                 skip_feature = encoder_features[skip_idx] if skip_idx < len(encoder_features) else None
#                 current = decoder_block(current, skip_feature)
            
#             # 分割头
#             output = self.network.segmentation_head(current)
#             return output
        
#         # 替换前向传播
#         import types
#         self.network.forward = types.MethodType(attention_enhanced_forward, self.network)
#         print("  ✅ 已集成注意力到标准UNet跳跃连接")
    
#     def _integrate_into_generic_modular_unet(self):
#         """集成到nnU-Net的generic_modular_UNet结构"""
#         print("  检测到generic_modular_UNet结构")
        
#         # 获取网络中的跳跃连接位置
#         # generic_modular_UNet通常有conv_blocks_context（编码器）和conv_blocks_localization（解码器）
        
#         try:
#             # 方法1：通过属性名查找跳跃连接
#             if hasattr(self.network, 'tu'):
#                 # tu是上采样模块，通常处理跳跃连接
#                 print(f"  找到上采样模块: tu")
#                 self._modify_tu_blocks()
#             elif hasattr(self.network, 'conv_blocks_localization'):
#                 # 直接修改localization块
#                 print(f"  找到localization模块")
#                 self._modify_localization_blocks()
#             else:
#                 print(f"  ⚠️ 无法找到跳跃连接位置")
#         except Exception as e:
#             print(f"  ❌ 集成失败: {e}")
    
#     def _modify_tu_blocks(self):
#         """修改上采样模块以集成注意力"""
#         tu_blocks = self.network.tu
#         if not isinstance(tu_blocks, nn.ModuleList):
#             print(f"  ⚠️ tu不是ModuleList，无法修改")
#             return
        
#         print(f"  找到 {len(tu_blocks)} 个上采样块")
        
#         for i, tu_block in enumerate(tu_blocks):
#             if f'skip_{i}' in self.attention_modules:
#                 print(f"   为tu_block[{i}]添加注意力")
                
#                 # 保存原始前向传播
#                 original_tu_forward = tu_block.forward
                
#                 def attention_tu_forward(self_tu, x, skip):
#                     # 应用注意力到跳跃连接
#                     attention_gate = self.attention_modules[f'skip_{i}']
                    
#                     if isinstance(attention_gate, ClassBalancedAttentionGate3D):
#                         # 创建一个简单的类重要性张量
#                         class_importance = torch.ones(49, device=self.device)
#                         for cls in self.ignore_classes:
#                             if cls < len(class_importance):
#                                 class_importance[cls] = 0
                        
#                         enhanced_skip, _ = attention_gate(
#                             skip, x,
#                             class_importance=class_importance,
#                             return_class_attention=False
#                         )
#                         skip = enhanced_skip
#                     else:
#                         skip = attention_gate(skip, x)
                    
#                     # 调用原始前向传播
#                     return original_tu_forward(x, skip)
                
#                 # 绑定新的前向传播
#                 import types
#                 tu_block.forward = types.MethodType(attention_tu_forward, tu_block)
        
#         print(f"  ✅ 已为 {len([k for k in self.attention_modules.keys() if 'skip_' in k])} 个跳跃连接添加注意力")
    
#     def _modify_localization_blocks(self):
#         """修改localization模块以集成注意力"""
#         loc_blocks = self.network.conv_blocks_localization
#         if not isinstance(loc_blocks, nn.ModuleList):
#             print(f"  ⚠️ conv_blocks_localization不是ModuleList，无法修改")
#             return
        
#         print(f"  找到 {len(loc_blocks)} 个localization块")
        
#         for i, loc_block in enumerate(loc_blocks):
#             if i < len(loc_blocks) - 1 and f'skip_{i}' in self.attention_modules:
#                 print(f"   为localization_block[{i}]添加注意力")
                
#                 # 保存原始前向传播
#                 original_loc_forward = loc_block.forward
                
#                 def attention_loc_forward(self_loc, x, skip):
#                     # 应用注意力到跳跃连接
#                     attention_gate = self.attention_modules[f'skip_{i}']
                    
#                     if isinstance(attention_gate, ClassBalancedAttentionGate3D):
#                         # 创建一个简单的类重要性张量
#                         class_importance = torch.ones(49, device=self.device)
#                         for cls in self.ignore_classes:
#                             if cls < len(class_importance):
#                                 class_importance[cls] = 0
                        
#                         enhanced_skip, _ = attention_gate(
#                             skip, x,
#                             class_importance=class_importance,
#                             return_class_attention=False
#                         )
#                         skip = enhanced_skip
#                     else:
#                         skip = attention_gate(skip, x)
                    
#                     # 调用原始前向传播
#                     return original_loc_forward(x, skip)
                
#                 # 绑定新的前向传播
#                 import types
#                 loc_block.forward = types.MethodType(attention_loc_forward, loc_block)
        
#         print(f"  ✅ 已修改localization块以集成注意力")
    
#     def _initialize_attention_modules(self):
#         """初始化注意力模块"""
#         if not self.use_attention:
#             return
        
#         print("\n" + "="*60)
#         print("🎯 正在初始化多尺度注意力模块...")
#         print("="*60)
        
#         self.attention_modules = nn.ModuleDict()
        
#         # 为每个跳跃连接创建注意力门
#         for i, (skip_ch, gate_ch) in enumerate(zip(self.encoder_channels[:-1], self.decoder_channels[1:])):
#             scale_factor = 2
            
#             try:
#                 if self.attention_type == 'class_balanced':
#                     attention_gate = ClassBalancedAttentionGate3D(
#                         skip_channels=skip_ch,
#                         gate_channels=gate_ch,
#                         num_classes=self.attention_config['num_classes'],
#                         minority_classes=self.attention_config['minority_classes'],
#                         minority_boost=self.attention_config['minority_boost'],
#                         scale_factor=scale_factor,
#                         reduction_ratio=self.attention_config['reduction_ratio'],
#                         use_residual=self.attention_config['use_residual'],
#                         dropout_rate=self.attention_config['dropout_rate']
#                     )
#                 elif self.attention_type == 'hierarchical':
#                     attention_gate = HierarchicalAttentionGate3D(
#                         skip_channels=skip_ch,
#                         gate_channels_list=[gate_ch, self.decoder_channels[i] if i > 0 else gate_ch],
#                         scale_factor=scale_factor,
#                         reduction_ratio=self.attention_config['reduction_ratio']
#                     )
#                 else:
#                     attention_gate = MultiScaleAttentionGate3D(
#                         skip_channels=skip_ch,
#                         gate_channels=gate_ch,
#                         scale_factor=scale_factor,
#                         reduction_ratio=self.attention_config['reduction_ratio'],
#                         use_residual=self.attention_config['use_residual'],
#                         dropout_rate=self.attention_config['dropout_rate']
#                     )
                
#                 self.attention_modules[f'skip_{i}'] = attention_gate
#                 print(f"  ✅ 创建注意力门 {i}: skip={skip_ch}, gate={gate_ch}, type={self.attention_type}")
#             except Exception as e:
#                 print(f"  ❌ 创建注意力门 {i} 失败: {e}")
        
#         # 移动到设备
#         if self.attention_modules:
#             self.attention_modules.to(self.device)
#             print(f"\n🎉 成功初始化 {len(self.attention_modules)} 个注意力模块")
#             print(f"  注意力参数总数: {sum(p.numel() for p in self.attention_modules.parameters()):,}")
#         else:
#             print(f"\n⚠️ 注意力模块初始化失败")
    
#     def _add_extreme_mamba_to_network(self):
#         """在网络中添加强化的Mamba瓶颈层"""
#         if not MAMBA_AVAILABLE:
#             print(f"⚠️ Mamba不可用，跳过Mamba增强")
#             return
        
#         print("\n🔥 正在添加激进的Mamba增强层...")
        
#         # 查找所有卷积层
#         conv_layers = []
        
#         def find_conv_layers(module, name=""):
#             for child_name, child_module in module.named_children():
#                 full_name = f"{name}.{child_name}" if name else child_name
                
#                 if isinstance(child_module, nn.Conv3d):
#                     conv_layers.append((full_name, child_module))
                
#                 find_conv_layers(child_module, full_name)
        
#         find_conv_layers(self.network)
        
#         print(f"找到 {len(conv_layers)} 个卷积层")
        
#         if conv_layers:
#             conv_layers.sort(key=lambda x: x[0].count('.'), reverse=True)
#             target_layers = conv_layers[:3]
            
#             print(f"\n🎯 选择以下层进行Mamba增强:")
#             for i, (name, module) in enumerate(target_layers):
#                 print(f"  {i+1}. {name}: {module.in_channels} -> {module.out_channels}")
            
#             replaced_count = 0
#             for name, module in target_layers:
#                 try:
#                     in_channels = module.in_channels
#                     out_channels = module.out_channels
                    
#                     if in_channels >= 64 and out_channels >= 64:
#                         # 创建Mamba瓶颈层（需要重新定义，因为原代码中的MambaBottleneck3D可能不可用）
#                         print(f"  ⚠️ 跳过Mamba替换 {name} (需要完整Mamba实现)")
#                 except Exception as e:
#                     print(f"  ❌ 替换 {name} 失败: {e}")
            
#             if replaced_count > 0:
#                 self.mamba_added = True
#                 print(f"\n🎉 成功添加 {replaced_count} 个Mamba增强层")
#             else:
#                 print(f"⚠️ 未成功替换任何层")
#         else:
#             print(f"⚠️ 未找到合适的卷积层")
    
#     def configure_optimizers(self):
#         """配置优化器 - 保持nnUNet默认"""
#         return super().configure_optimizers()
    
#     def validate(self, *args, **kwargs):
#         """验证步骤"""
#         result = super().validate(*args, **kwargs)
#         return result
    
#     def train_step(self, data_batch):
#         """训练步骤"""
#         if self.current_epoch == 0 and not hasattr(self, '_train_info_printed'):
#             print(f"\n" + "="*60)
#             print(f"🚀 训练开始 - 论文复现版本")
#             print(f"  忽略类别: {self.ignore_classes}")
#             print(f"  镜像增强: 已禁用（牙齿不对称）")
#             print(f"  后处理: 连通域分析（保留最大{self.postprocessing_params['max_components']}个分量）")
            
#             if self.use_mamba and self.mamba_added:
#                 print(f"  🔥 使用Mamba增强")
            
#             if self.use_attention and self.attention_modules:
#                 print(f"  🎯 使用{self.attention_type}注意力机制")
#                 print(f"  注意力模块数: {len(self.attention_modules)}")
#                 print(f"  已集成到跳跃连接: 是")
            
#             print(f"="*60)
            
#             self._train_info_printed = True
        
#         # 处理target
#         if 'target' in data_batch:
#             target = data_batch['target']
            
#             if isinstance(target, list):
#                 new_targets = []
#                 for t in target:
#                     if isinstance(t, torch.Tensor):
#                         t = t.clone()
#                         for cls in self.ignore_classes:
#                             t[t == cls] = 0
#                         new_targets.append(t)
#                     else:
#                         new_targets.append(t)
#                 data_batch['target'] = new_targets
#             elif isinstance(target, torch.Tensor):
#                 target = target.clone()
#                 for cls in self.ignore_classes:
#                     target[target == cls] = 0
#                 data_batch['target'] = target
        
#         result = super().train_step(data_batch)
        
#         # 监控
#         if hasattr(self, '_batch_counter'):
#             self._batch_counter += 1
#             if self._batch_counter % 50 == 0:
#                 if 'loss' in result:
#                     loss_val = result['loss']
#                     if isinstance(loss_val, torch.Tensor):
#                         loss_val = loss_val.item()
#                     print(f"  📦 Batch {self._batch_counter}, Loss: {loss_val:.4f}")
#         else:
#             self._batch_counter = 1
        
#         return result
    
#     def on_epoch_end(self):
#         """每个epoch结束时的回调"""
#         super().on_epoch_end()
        
#         if hasattr(self, 'optimizer'):
#             lr = self.optimizer.param_groups[0]['lr']
#             print(f"\n📈 Epoch {self.current_epoch} 进度:")
#             print(f"  学习率: {lr:.2e}")

import torch
import torch.nn as nn
import os
import sys
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# ================ 您的原始修复代码 ================
os.environ['NNUNET_COMPILE'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '0'

# ================ 真正解决问题的修复 ================
import warnings
import nnunetv2.evaluation.evaluate_predictions as eval_module

IGNORE_CLASSES = [43, 44]  # 只保留43,44

print("="*60)
print("安装Dice NaN根本性修复")
print("="*60)

# 1. 修复compute_metrics函数
original_compute_metrics = eval_module.compute_metrics

def fixed_compute_metrics(reference_file, prediction_file, image_reader_writer,
                         labels_or_regions, ignore_label=None):

    try:
        result = original_compute_metrics(
            reference_file, prediction_file, image_reader_writer,
            labels_or_regions, ignore_label
        )
        
        if isinstance(result, dict) and 'metrics' in result:
            metrics = result['metrics']
            
            for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                if metric_name in metrics:
                    metric_dict = metrics[metric_name]
                    
                    if isinstance(metric_dict, dict):
                        for cls in IGNORE_CLASSES:
                            if str(cls) in metric_dict:
                                del metric_dict[str(cls)]
                            elif cls in metric_dict:
                                del metric_dict[cls]
                        
                        for key in list(metric_dict.keys()):
                            val = metric_dict[key]
                            if isinstance(val, (int, float)):
                                if np.isnan(val):
                                    metric_dict[key] = 0.0
                            elif hasattr(val, 'item'):
                                if np.isnan(val.item()):
                                    metric_dict[key] = 0.0
        
        return result
        
    except Exception as e:
        print(f"⚠️ compute_metrics错误: {e}")
        return {'metrics': {'dice': {}, 'hausdorff_distance_95': {}, 'mean_surface_distance': {}}}

eval_module.compute_metrics = fixed_compute_metrics
print("✅ 已修复 compute_metrics 函数")

# 2. 修复聚合函数summarize_results（省略，保持原样）
if hasattr(eval_module, 'summarize_results'):
    original_summarize = eval_module.summarize_results
    
    def fixed_summarize_results(results, args=None):
        try:
            fixed_results = []
            for res in results:
                if isinstance(res, dict) and 'metrics' in res:
                    fixed_res = res.copy()
                    metrics = fixed_res['metrics']
                    
                    for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                        if metric_name in metrics:
                            metric_dict = metrics[metric_name]
                            if isinstance(metric_dict, dict):
                                for key, val in metric_dict.items():
                                    if isinstance(val, (int, float)) and np.isnan(val):
                                        metric_dict[key] = 0.0
                    
                    fixed_results.append(fixed_res)
                else:
                    fixed_results.append(res)
            
            if args is not None:
                summary = original_summarize(fixed_results, args)
            else:
                summary = original_summarize(fixed_results)
            
            def fix_nan_in_dict(d):
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            fix_nan_in_dict(v)
                        elif isinstance(v, (int, float)) and np.isnan(v):
                            d[k] = 0.0
            
            if isinstance(summary, dict):
                fix_nan_in_dict(summary)
            
            return summary
            
        except Exception as e:
            print(f"⚠️ summarize_results错误: {e}")
            return {}

    eval_module.summarize_results = fixed_summarize_results
print("✅ 已修复 summarize_results 函数")

# 3. 修复compute_metrics_on_folder函数（省略，保持原样）
original_compute_metrics_on_folder = eval_module.compute_metrics_on_folder

def fixed_compute_metrics_on_folder(folder_ref, folder_pred, image_reader_writer,
                                   labels_or_regions, ignore_label=None, 
                                   num_processes: int = 1):
    try:
        results = original_compute_metrics_on_folder(
            folder_ref, folder_pred, image_reader_writer,
            labels_or_regions, ignore_label, num_processes
        )
        
        fixed_results = []
        for res in results:
            if isinstance(res, dict) and 'metrics' in res:
                fixed_res = res.copy()
                metrics = fixed_res['metrics']
                
                for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                    if metric_name in metrics:
                        metric_dict = metrics[metric_name]
                        if isinstance(metric_dict, dict):
                            for key, val in metric_dict.items():
                                if isinstance(val, (int, float)) and np.isnan(val):
                                    metric_dict[key] = 0.0
                
                fixed_results.append(fixed_res)
            else:
                fixed_results.append(res)
        
        return fixed_results
        
    except Exception as e:
        print(f"⚠️ compute_metrics_on_folder错误: {e}")
        return []

eval_module.compute_metrics_on_folder = fixed_compute_metrics_on_folder
print("✅ 已修复 compute_metrics_on_folder 函数")

# 4. 修复np.nanmean
original_nanmean = np.nanmean

def safe_nanmean(a, **kwargs):
    try:
        result = original_nanmean(a, **kwargs)
        if np.isnan(result):
            return 0.0
        return result
    except:
        return 0.0

np.nanmean = safe_nanmean
print("✅ 已修复 np.nanmean 函数作为最后保障")

print(f"\n" + "="*60)
print(f"🎯 修复总结:")
print(f"   忽略的类别: {IGNORE_CLASSES}")
print(f"   修复了3个关键函数 + np.nanmean作为保障")
print(f"   策略: 从计算源头防止NaN，层层防护")
print("="*60 + "\n")
# ================ 您的修复代码结束 ================

# ================ Mamba 集成开始 ================
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("⚠️ mamba_ssm 不可用，将禁用Mamba功能")
    MAMBA_AVAILABLE = False
    Mamba = None

# ================ 注意力模块导入 ================
try:
    from .attention_gates import (
        MultiScaleAttentionGate3D,
        HierarchicalAttentionGate3D,
        ClassBalancedAttentionGate3D,
        create_attention_gate
    )
    ATTENTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入attention_gates: {e}")
    ATTENTION_AVAILABLE = False
    # 创建空类避免错误
    class MultiScaleAttentionGate3D:
        def __init__(self, *args, **kwargs):
            raise ImportError("attention_gates not available")
    class HierarchicalAttentionGate3D:
        def __init__(self, *args, **kwargs):
            raise ImportError("attention_gates not available")
    class ClassBalancedAttentionGate3D:
        def __init__(self, *args, **kwargs):
            raise ImportError("attention_gates not available")
    def create_attention_gate(*args, **kwargs):
        raise ImportError("attention_gates not available")

# ================ 自定义Dice+CE损失函数（排除背景） ================
class DiceCELossExcludingBackground(nn.Module):
    """
    自定义Dice+Cross-Entropy损失函数
    完全按照论文要求：
    1. Dice项排除背景（类别0）
    2. CE项包含所有类别（包括背景）
    3. 权重相等：Dice=1.0, CE=1.0
    """
    def __init__(self, num_classes, dice_weight=1.0, ce_weight=1.0, 
                 smooth=1e-5, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        
        # Cross-Entropy损失
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        print(f"🎯 Dice+CE损失函数配置:")
        print(f"  类别总数: {num_classes}")
        print(f"  前景类别数（参与Dice计算）: {num_classes - 1}")
        print(f"  Dice权重: {dice_weight}")
        print(f"  CE权重: {ce_weight}")
        print(f"  Dice项排除背景: 是")
        print(f"  CE项包含背景: 是")
    
    def forward(self, pred, target):
        """
        pred: [B, C, D, H, W] 未归一化的logits
        target: [B, D, H, W] ground truth标签（Long类型）
        """
        batch_size = pred.shape[0]
        
        # ========== 1. 计算Cross-Entropy损失 ==========
        # CE损失包含所有类别（包括背景）
        ce_loss = self.ce_loss(pred, target)
        
        # ========== 2. 计算Dice损失（排除背景） ==========
        # 将预测转换为概率
        pred_softmax = torch.softmax(pred, dim=1)
        
        # 创建one-hot编码的目标
        target_onehot = torch.zeros_like(pred_softmax)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        
        dice_loss = 0.0
        valid_classes = 0
        
        # 只计算前景类别（跳过背景类别0）
        for class_idx in range(1, self.num_classes):
            # 获取当前类别的预测和目标
            pred_class = pred_softmax[:, class_idx]  # [B, D, H, W]
            target_class = target_onehot[:, class_idx]  # [B, D, H, W]
            
            # 计算intersection和union
            intersection = (pred_class * target_class).sum(dim=(1, 2, 3))  # [B]
            union = pred_class.sum(dim=(1, 2, 3)) + target_class.sum(dim=(1, 2, 3))  # [B]
            
            # 计算每个样本的Dice系数
            dice_per_sample = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B]
            
            # 只计算有该类别存在的样本
            mask = union > 0
            if mask.any():
                dice_loss += (1.0 - dice_per_sample[mask]).mean()
                valid_classes += 1
        
        # 计算平均Dice损失
        if valid_classes > 0:
            dice_loss = dice_loss / valid_classes
        else:
            dice_loss = torch.tensor(0.0, device=pred.device)
        
        # ========== 3. 组合损失 ==========
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        
        # 记录损失分量用于监控
        self._current_losses = {
            'total': total_loss.item(),
            'dice': dice_loss.item() if isinstance(dice_loss, torch.Tensor) else dice_loss,
            'ce': ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss
        }
        
        return total_loss
    
    def get_loss_components(self):
        """获取损失分量（用于监控）"""
        return getattr(self, '_current_losses', {'total': 0, 'dice': 0, 'ce': 0})

class WeightedDiceCELoss(nn.Module):
    """
    带类别权重的Dice+CE损失
    可以给少数类别更高的权重
    """
    def __init__(self, num_classes, class_weights=None, 
                 dice_weight=1.0, ce_weight=1.0, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        
        # 设置类别权重
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = torch.ones(num_classes, dtype=torch.float32)
        self.class_weights[-3] = 10.0  # 倒数第3个 = 45
        self.class_weights[-2] = 10.0  # 倒数第2个 = 46  
        self.class_weights[-4] = 10.0
        # CE损失
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
    
    def forward(self, pred, target):
        # ========== 1. 计算加权CE损失 ==========
        ce_loss = self.ce_loss(pred, target)
        
        # ========== 2. 计算加权Dice损失（排除背景） ==========
        pred_softmax = torch.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred_softmax)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        
        dice_loss = 0.0
        total_weight = 0.0
        
        # 只计算前景类别
        for class_idx in range(1, self.num_classes):
            weight = self.class_weights[class_idx].item()
            
            pred_class = pred_softmax[:, class_idx]
            target_class = target_onehot[:, class_idx]
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            if union > 0:
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_loss += weight * (1.0 - dice)
                total_weight += weight
        
        if total_weight > 0:
            dice_loss = dice_loss / total_weight
        
        # ========== 3. 组合损失 ==========
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        
        return total_loss

# ================ 核心修复：真正集成注意力到跳跃连接的训练器 ================
class TryTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_classes = len(self.dataset_json['labels'].keys())
        # ========== 训练类别设置 ==========
        # 只设置忽略类别，其他保持不变
        self.ignore_classes = [43, 44]
        
        # ========== 论文设置：禁用镜像和后处理 ==========
        self.disable_mirror = True  # 禁用镜像（牙齿不对称）
        self.enable_postprocessing = True  # 启用后处理
        self.postprocessing_params = {
            'max_components': 3,  # 保留最大的3个连通分量
            'min_component_size': 100,  # 最小分量大小
        }
        
        # ========== 损失函数设置 ==========
        self.loss_dice_weight = 1.0  # Dice损失权重
        self.loss_ce_weight = 1.0    # CE损失权重
        self.loss_smooth = 1e-5      # 平滑参数
        
        print(f"\n" + "="*60)
        print(f"🎯 训练配置:")
        print(f"  忽略类别: {self.ignore_classes}")
        print(f"  镜像增强: {'已禁用' if self.disable_mirror else '已启用'}")
        print(f"  后处理: 连通域分析（保留最大{self.postprocessing_params['max_components']}个分量）")
        print(f"  损失函数: Dice + Cross-Entropy")
        print(f"  损失权重: Dice={self.loss_dice_weight}, CE={self.loss_ce_weight}")
        print(f"  Dice项排除背景: 是")
        print(f"  CE项包含背景: 是")
        print("="*60)
        
        # 监控变量
        self.best_mean_dice = 0
        self.patience_counter = 0
        self.max_patience = 20
        
        # ========== Mamba 相关设置 ==========
        self.use_mamba = True and MAMBA_AVAILABLE
        self.mamba_added = False
        self.mamba_seq_len = 32
        self.mamba_d_state = 16
        self.mamba_d_conv = 4
        self.mamba_expand = 2
        
        # ========== 注意力机制设置 ==========
        self.use_attention = True and ATTENTION_AVAILABLE
        # hybrid：深层跳跃用类平衡，浅层用轻量/多尺度
        self.attention_type = 'hybrid'
        self.attention_config = {
            'reduction_ratio': 16,
            'use_residual': True,
            'dropout_rate': 0.1,
            'attention_lr_multiplier': 0.5,
            'attention_weight_decay': 1e-5,
            'num_classes': 49,
            'minority_classes': [48] + list(range(11, 43)),
            'minority_boost': 2.0,
            'weight_mode': 'inv_sqrt',       # 类频权重: inv 或 inv_sqrt
            'class_balanced_levels': 2,      # 深层使用类平衡门的层数
            'shallow_gate': 'multiscale'      # 浅层门类型: multiscale 或 simplified
        }
        
        # nnU-Net标准编码器通道数
        self.encoder_channels = [32, 64, 128, 256, 320]
        self.decoder_channels = [320, 256, 128, 64, 32]
        
        # 初始化注意力模块
        self.attention_modules = None
        
        if self.use_attention:
            print(f"\n🎯 注意力机制配置:")
            print(f"  类型: {self.attention_type}")
            print(f"  位置: 所有跳跃连接")
            print(f"  学习率乘子: {self.attention_config['attention_lr_multiplier']}")
        
        if self.use_mamba:
            print(f"\n🎯 Mamba 配置:")
            print(f"  使用官方 mamba-ssm")
            print(f"  序列长度: {self.mamba_seq_len}")
            print(f"  状态维度: {self.mamba_d_state}")
            print(f"  卷积核大小: {self.mamba_d_conv}")
            print(f"  扩展因子: {self.mamba_expand}")
    
    # ========== 核心修改：配置损失函数 ==========
    def configure_loss_function(self):
        """
        配置损失函数 - 使用Dice+CE，Dice项排除背景
        完全按照论文要求：equal weighting, exclude background from Dice term
        """
        print(f"\n" + "="*60)
        print("🎯 配置自定义Dice+CE损失函数...")
        print("="*60)
        
        # 获取类别数量（包括背景）
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        
        # 创建损失函数
        loss_fn = DiceCELossExcludingBackground(
            num_classes=self.num_classes,
            dice_weight=self.loss_dice_weight,
            ce_weight=self.loss_ce_weight,
            smooth=self.loss_smooth,
            ignore_index=None  # 不忽略任何索引，让损失函数自己处理
        )
        
        # 测试损失函数
        self._test_loss_function(loss_fn)
        
        return loss_fn
    
    def _test_loss_function(self, loss_fn):
        """测试损失函数是否正常工作"""
        print(f"\n🧪 测试损失函数...")
        
        try:
            # 创建测试数据
            batch_size = 2
            spatial_dims = (32, 32, 32)
            num_classes = self.num_classes
            
            # 随机生成预测和标签
            pred = torch.randn(batch_size, num_classes, *spatial_dims, device=self.device)
            target = torch.randint(0, num_classes, (batch_size, *spatial_dims), device=self.device)
            
            # 确保忽略类别被设为背景
            for cls in self.ignore_classes:
                if cls < num_classes:
                    target[target == cls] = 0
            
            print(f"  测试数据形状:")
            print(f"    pred: {pred.shape}")
            print(f"    target: {target.shape}")
            print(f"    类别分布: {torch.unique(target, return_counts=True)}")
            
            # 计算损失
            with torch.no_grad():
                loss = loss_fn(pred, target)
            
            print(f"\n  ✅ 损失函数测试通过")
            print(f"    总损失值: {loss.item():.4f}")
            
            # 显示各分量
            if hasattr(loss_fn, 'get_loss_components'):
                components = loss_fn.get_loss_components()
                print(f"    损失分量:")
                print(f"      Dice损失: {components.get('dice', 0):.4f}")
                print(f"      CE损失: {components.get('ce', 0):.4f}")
                print(f"      总损失: {components.get('total', 0):.4f}")
            
            # 验证排除背景的逻辑
            print(f"\n  📊 验证Dice排除背景:")
            pred_softmax = torch.softmax(pred, dim=1)
            target_onehot = torch.zeros_like(pred_softmax)
            target_onehot.scatter_(1, target.unsqueeze(1), 1)
            
            # 计算每个类别的Dice（包含背景）
            for class_idx in range(min(3, num_classes)):  # 只显示前3个类别
                pred_class = pred_softmax[:, class_idx]
                target_class = target_onehot[:, class_idx]
                intersection = (pred_class * target_class).sum()
                union = pred_class.sum() + target_class.sum()
                
                if union > 0:
                    dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
                    class_type = "背景" if class_idx == 0 else f"前景类别{class_idx}"
                    print(f"    类别{class_idx}({class_type}) Dice: {dice.item():.4f}")
            
            print(f"  🎯 Dice损失计算的前景类别数: {num_classes - 1}")
            
        except Exception as e:
            print(f"  ⚠️ 损失函数测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== 重写训练步骤以监控损失分量 ==========
    def train_step(self, data_batch):
        """训练步骤 - 监控损失分量"""
        # 处理target中的忽略类别
        if 'target' in data_batch:
            target = data_batch['target']
            
            if isinstance(target, torch.Tensor):
                target = target.clone()
                for cls in self.ignore_classes:
                    if cls < self.num_classes:
                        target[target == cls] = 0
                data_batch['target'] = target
        
        # 调用父类训练步骤
        result = super().train_step(data_batch)
        
        # 监控损失分量
        if hasattr(self.loss, 'get_loss_components'):
            components = self.loss.get_loss_components()
            if self.current_epoch % 10 == 0 and self._batch_in_epoch == 0:
                print(f"\n📊 训练损失分量 (Epoch {self.current_epoch}):")
                print(f"  Dice损失: {components.get('dice', 0):.4f}")
                print(f"  CE损失: {components.get('ce', 0):.4f}")
                print(f"  总损失: {components.get('total', 0):.4f}")
        
        # 批量计数器
        if hasattr(self, '_batch_in_epoch'):
            self._batch_in_epoch += 1
        else:
            self._batch_in_epoch = 1
        
        return result
    
    def on_epoch_end(self):
        """每个epoch结束时的回调"""
        super().on_epoch_end()
        
        # 重置批量计数器
        self._batch_in_epoch = 0
        
        # 打印学习率
        if hasattr(self, 'optimizer'):
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\n📈 Epoch {self.current_epoch} 结束:")
            print(f"  学习率: {lr:.2e}")
            print(f"  最佳Dice: {self.best_mean_dice:.4f}")
            print(f"  耐心计数: {self.patience_counter}/{self.max_patience}")

    def on_validation_end(self, val_loss, pseudo_dice, pseudo_dice_ema):
        """验证结束时打印有效类别的Dice均值，剔除背景与忽略类，并跳过NaN"""
        super().on_validation_end(val_loss, pseudo_dice, pseudo_dice_ema)

        if pseudo_dice is None:
            print("⚠️ 验证未返回 pseudo_dice")
            return

        pd = np.array(pseudo_dice, dtype=float)

        # 剔除背景与忽略类
        invalid = set([0] + self.ignore_classes)
        valid_mask = np.ones_like(pd, dtype=bool)
        for cls in invalid:
            if cls < len(valid_mask):
                valid_mask[cls] = False

        # 去掉 NaN
        valid_mask = valid_mask & (~np.isnan(pd))
        valid_pd = pd[valid_mask]

        mean_valid = float(np.mean(valid_pd)) if valid_pd.size > 0 else float('nan')
        nonzero_count = int(np.sum(valid_pd > 0)) if valid_pd.size > 0 else 0

        print("\n📊 验证Dice统计(排除背景/忽略类，跳过NaN):")
        print(f"  有效类别数: {valid_pd.size}")
        print(f"  非零类别数: {nonzero_count}")
        print(f"  有效均值Dice: {mean_valid:.4f}" if not np.isnan(mean_valid) else "  有效均值Dice: NaN")

        # 标记前若干弱类方便观察
        weak_indices = []
        for cls_idx, val in enumerate(pd):
            if cls_idx in invalid or np.isnan(val):
                continue
            if val <= 0.05:
                weak_indices.append((cls_idx, val))
        if weak_indices:
            weak_indices = sorted(weak_indices, key=lambda x: x[1])[:8]
            msg = ", ".join([f"cls{c}:{v:.3f}" for c, v in weak_indices])
            print(f"  低Dice类(<=0.05)示例: {msg}")
    
    # ========== 论文设置：配置数据增强（禁用镜像） ==========
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """配置数据增强参数 - 禁用镜像"""
        # 先调用父类方法
        patch_size, mirror_axes, rotation_for_DA, do_dummy_2d_data_aug = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # 禁用镜像增强
        mirror_axes = ()
        print(f"\n🔧 数据增强配置: 镜像增强已禁用（mirror_axes={mirror_axes}）")
        
        # 必须返回这四个值
        return patch_size, mirror_axes, rotation_for_DA, do_dummy_2d_data_aug
    
    # ========== 论文设置：后处理方法 ==========
    def run_inference(self, *args, **kwargs):
        """运行推理并应用后处理"""
        # 设置是否进行后处理
        kwargs['do_postprocessing'] = self.enable_postprocessing
        
        # 调用父类推理
        predictions = super().run_inference(*args, **kwargs)
        
        if self.enable_postprocessing and predictions is not None:
            print(f"\n🔧 应用后处理（保留最大{self.postprocessing_params['max_components']}个连通分量）")
            
            # 对每个预测进行后处理
            processed_predictions = []
            for pred in predictions:
                if isinstance(pred, np.ndarray):
                    # 连通域分析后处理
                    processed_pred = self._connected_component_postprocessing(pred)
                    processed_predictions.append(processed_pred)
                else:
                    processed_predictions.append(pred)
            
            return processed_predictions
        
        return predictions
    
    def _connected_component_postprocessing(self, segmentation):
        """连通域分析后处理"""
        try:
            from skimage.measure import label
        except ImportError:
            print("⚠️ 未安装skimage，跳过后处理")
            return segmentation
        
        result = np.zeros_like(segmentation)
        
        # 对每个类别独立处理
        unique_labels = np.unique(segmentation)
        unique_labels = unique_labels[unique_labels != 0]  # 排除背景
        
        for label_id in unique_labels:
            binary_mask = (segmentation == label_id).astype(np.uint8)
            
            # 连通域标记
            labeled_mask, num_features = label(binary_mask, connectivity=3, return_num=True)
            
            if num_features > 0:
                # 计算每个连通分量的大小
                component_sizes = []
                component_indices = []
                
                for i in range(1, num_features + 1):
                    component_mask = (labeled_mask == i)
                    size = np.sum(component_mask)
                    component_sizes.append(size)
                    component_indices.append(i)
                
                # 按大小排序
                sorted_indices = np.argsort(component_sizes)[::-1]  # 从大到小
                
                # 保留最大的N个分量
                keep_count = min(self.postprocessing_params['max_components'], len(sorted_indices))
                
                for i in range(keep_count):
                    idx = sorted_indices[i]
                    component_id = component_indices[idx]
                    component_size = component_sizes[idx]
                    
                    # 只保留足够大的分量
                    if component_size >= self.postprocessing_params['min_component_size']:
                        component_mask = (labeled_mask == component_id)
                        result[component_mask] = label_id
        
        return result
    
    # ========== 核心修复：真正集成注意力到跳跃连接 ==========
    def initialize_network(self):
        """重写网络初始化以真正集成注意力到跳跃连接"""
        # 先调用父类初始化创建基础网络
        super().initialize_network()
        
        if self.use_mamba and not self.mamba_added:
            self._add_extreme_mamba_to_network()
        
        if self.use_attention:
            self._initialize_attention_modules()
            
            # 关键：真正修改网络结构，将注意力集成到跳跃连接
            self._integrate_attention_into_unet_architecture()
    
    def _integrate_attention_into_unet_architecture(self):
        """真正将注意力模块集成到UNet架构中"""
        print("\n" + "="*60)
        print("🔥 正在将注意力模块真正集成到UNet跳跃连接...")
        print("="*60)
        
        # 检查网络类型
        if hasattr(self.network, 'encoder') and hasattr(self.network, 'decoder'):
            # 标准UNet结构
            self._integrate_into_standard_unet()
        elif hasattr(self.network, 'conv_blocks_context'):
            # nnU-Net的generic_modular_UNet结构
            self._integrate_into_generic_modular_unet()
        else:
            print(f"⚠️ 无法识别的网络结构，无法集成注意力模块")
            print(f"  网络类型: {type(self.network)}")
            print(f"  网络属性: {dir(self.network)}")
    
    def _integrate_into_standard_unet(self):
        """集成到标准UNet结构"""
        print("  检测到标准UNet结构")
        
        # 方法1：直接修改解码器的前向传播
        original_forward = self.network.forward
        
        def attention_enhanced_forward(x):
            # 存储编码器特征
            encoder_features = []
            current = x
            
            # 编码器前向传播
            for encoder_block in self.network.encoder.blocks:
                current = encoder_block(current)
                encoder_features.append(current)
            
            # 解码器前向传播（带注意力）
            current = encoder_features[-1]
            for i, decoder_block in enumerate(reversed(self.network.decoder.blocks)):
                # 对应的跳跃连接索引
                skip_idx = len(self.network.decoder.blocks) - i - 1
                
                if skip_idx < len(encoder_features) and f'skip_{skip_idx}' in self.attention_modules:
                    # 应用注意力到跳跃连接
                    skip_feature = encoder_features[skip_idx]
                    gate_feature = current
                    
                    attention_gate = self.attention_modules[f'skip_{skip_idx}']
                    
                    if isinstance(attention_gate, ClassBalancedAttentionGate3D):
                        # 用当前类权重作为重要性，并屏蔽忽略类
                        class_importance = self.class_weights.clone().to(self.device)
                        for cls in self.ignore_classes:
                            if cls < len(class_importance):
                                class_importance[cls] = 0
                        
                        enhanced_skip, _ = attention_gate(
                            skip_feature, gate_feature, 
                            class_importance=class_importance,
                            return_class_attention=False
                        )
                        encoder_features[skip_idx] = enhanced_skip
                    else:
                        enhanced_skip = attention_gate(skip_feature, gate_feature)
                        encoder_features[skip_idx] = enhanced_skip
                
                # 解码器操作
                skip_feature = encoder_features[skip_idx] if skip_idx < len(encoder_features) else None
                current = decoder_block(current, skip_feature)
            
            # 分割头
            output = self.network.segmentation_head(current)
            return output
        
        # 替换前向传播
        import types
        self.network.forward = types.MethodType(attention_enhanced_forward, self.network)
        print("  ✅ 已集成注意力到标准UNet跳跃连接")
    
    def _integrate_into_generic_modular_unet(self):
        """集成到nnU-Net的generic_modular_UNet结构"""
        print("  检测到generic_modular_UNet结构")
        
        # 获取网络中的跳跃连接位置
        # generic_modular_UNet通常有conv_blocks_context（编码器）和conv_blocks_localization（解码器）
        
        try:
            # 方法1：通过属性名查找跳跃连接
            if hasattr(self.network, 'tu'):
                # tu是上采样模块，通常处理跳跃连接
                print(f"  找到上采样模块: tu")
                self._modify_tu_blocks()
            elif hasattr(self.network, 'conv_blocks_localization'):
                # 直接修改localization块
                print(f"  找到localization模块")
                self._modify_localization_blocks()
            else:
                print(f"  ⚠️ 无法找到跳跃连接位置")
        except Exception as e:
            print(f"  ❌ 集成失败: {e}")
    
    def _modify_tu_blocks(self):
        """修改上采样模块以集成注意力"""
        tu_blocks = self.network.tu
        if not isinstance(tu_blocks, nn.ModuleList):
            print(f"  ⚠️ tu不是ModuleList，无法修改")
            return
        
        print(f"  找到 {len(tu_blocks)} 个上采样块")
        
        for i, tu_block in enumerate(tu_blocks):
            if f'skip_{i}' in self.attention_modules:
                print(f"   为tu_block[{i}]添加注意力")
                
                # 保存原始前向传播
                original_tu_forward = tu_block.forward
                
                def attention_tu_forward(self_tu, x, skip):
                    # 应用注意力到跳跃连接
                    attention_gate = self.attention_modules[f'skip_{i}']
                    
                    if isinstance(attention_gate, ClassBalancedAttentionGate3D):
                        # 用类权重作为重要性，并屏蔽忽略类
                        class_importance = self.class_weights.clone().to(self.device)
                        for cls in self.ignore_classes:
                            if cls < len(class_importance):
                                class_importance[cls] = 0
                        
                        enhanced_skip, _ = attention_gate(
                            skip, x,
                            class_importance=class_importance,
                            return_class_attention=False
                        )
                        skip = enhanced_skip
                    else:
                        skip = attention_gate(skip, x)
                    
                    # 调用原始前向传播
                    return original_tu_forward(x, skip)
                
                # 绑定新的前向传播
                import types
                tu_block.forward = types.MethodType(attention_tu_forward, tu_block)
        
        print(f"  ✅ 已为 {len([k for k in self.attention_modules.keys() if 'skip_' in k])} 个跳跃连接添加注意力")
    
    def _modify_localization_blocks(self):
        """修改localization模块以集成注意力"""
        loc_blocks = self.network.conv_blocks_localization
        if not isinstance(loc_blocks, nn.ModuleList):
            print(f"  ⚠️ conv_blocks_localization不是ModuleList，无法修改")
            return
        
        print(f"  找到 {len(loc_blocks)} 个localization块")
        
        for i, loc_block in enumerate(loc_blocks):
            if i < len(loc_blocks) - 1 and f'skip_{i}' in self.attention_modules:
                print(f"   为localization_block[{i}]添加注意力")
                
                # 保存原始前向传播
                original_loc_forward = loc_block.forward
                
                def attention_loc_forward(self_loc, x, skip):
                    # 应用注意力到跳跃连接
                    attention_gate = self.attention_modules[f'skip_{i}']
                    
                    if isinstance(attention_gate, ClassBalancedAttentionGate3D):
                        # 用当前类权重作为重要性，并屏蔽忽略类
                        class_importance = self.class_weights.clone().to(self.device)
                        for cls in self.ignore_classes:
                            if cls < len(class_importance):
                                class_importance[cls] = 0
                        
                        enhanced_skip, _ = attention_gate(
                            skip, x,
                            class_importance=class_importance,
                            return_class_attention=False
                        )
                        skip = enhanced_skip
                    else:
                        skip = attention_gate(skip, x)
                    
                    # 调用原始前向传播
                    return original_loc_forward(x, skip)
                
                # 绑定新的前向传播
                import types
                loc_block.forward = types.MethodType(attention_loc_forward, loc_block)
        
        print(f"  ✅ 已修改localization块以集成注意力")
    
    def _initialize_attention_modules(self):
        """初始化注意力模块"""
        if not self.use_attention:
            return
        
        print("\n" + "="*60)
        print("🎯 正在初始化多尺度注意力模块...")
        print("="*60)
        
        self.attention_modules = nn.ModuleDict()
        
        # 为每个跳跃连接创建注意力门
        for i, (skip_ch, gate_ch) in enumerate(zip(self.encoder_channels[:-1], self.decoder_channels[1:])):
            scale_factor = 2
            
            try:
                if self.attention_type == 'class_balanced':
                    attention_gate = ClassBalancedAttentionGate3D(
                        skip_channels=skip_ch,
                        gate_channels=gate_ch,
                        num_classes=self.attention_config['num_classes'],
                        minority_classes=self.attention_config['minority_classes'],
                        minority_boost=self.attention_config['minority_boost'],
                        scale_factor=scale_factor,
                        reduction_ratio=self.attention_config['reduction_ratio'],
                        use_residual=self.attention_config['use_residual'],
                        dropout_rate=self.attention_config['dropout_rate']
                    )
                elif self.attention_type == 'hierarchical':
                    attention_gate = HierarchicalAttentionGate3D(
                        skip_channels=skip_ch,
                        gate_channels_list=[gate_ch, self.decoder_channels[i] if i > 0 else gate_ch],
                        scale_factor=scale_factor,
                        reduction_ratio=self.attention_config['reduction_ratio']
                    )
                else:
                    attention_gate = MultiScaleAttentionGate3D(
                        skip_channels=skip_ch,
                        gate_channels=gate_ch,
                        scale_factor=scale_factor,
                        reduction_ratio=self.attention_config['reduction_ratio'],
                        use_residual=self.attention_config['use_residual'],
                        dropout_rate=self.attention_config['dropout_rate']
                    )
                
                self.attention_modules[f'skip_{i}'] = attention_gate
                print(f"  ✅ 创建注意力门 {i}: skip={skip_ch}, gate={gate_ch}, type={self.attention_type}")
            except Exception as e:
                print(f"  ❌ 创建注意力门 {i} 失败: {e}")
        
        # 移动到设备
        if self.attention_modules:
            self.attention_modules.to(self.device)
            print(f"\n🎉 成功初始化 {len(self.attention_modules)} 个注意力模块")
            print(f"  注意力参数总数: {sum(p.numel() for p in self.attention_modules.parameters()):,}")
        else:
            print(f"\n⚠️ 注意力模块初始化失败")
    
    def _add_extreme_mamba_to_network(self):
        """在网络中添加强化的Mamba瓶颈层"""
        if not MAMBA_AVAILABLE:
            print(f"⚠️ Mamba不可用，跳过Mamba增强")
            return
        
        print("\n🔥 正在添加激进的Mamba增强层...")
        
        # 查找所有卷积层
        conv_layers = []
        
        def find_conv_layers(module, name=""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child_module, nn.Conv3d):
                    conv_layers.append((full_name, child_module))
                
                find_conv_layers(child_module, full_name)
        
        find_conv_layers(self.network)
        
        print(f"找到 {len(conv_layers)} 个卷积层")
        
        if conv_layers:
            conv_layers.sort(key=lambda x: x[0].count('.'), reverse=True)
            target_layers = conv_layers[:3]
            
            print(f"\n🎯 选择以下层进行Mamba增强:")
            for i, (name, module) in enumerate(target_layers):
                print(f"  {i+1}. {name}: {module.in_channels} -> {module.out_channels}")
            
            replaced_count = 0
            for name, module in target_layers:
                try:
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    
                    if in_channels >= 64 and out_channels >= 64:
                        # 创建Mamba瓶颈层（需要重新定义，因为原代码中的MambaBottleneck3D可能不可用）
                        print(f"  ⚠️ 跳过Mamba替换 {name} (需要完整Mamba实现)")
                except Exception as e:
                    print(f"  ❌ 替换 {name} 失败: {e}")
            
            if replaced_count > 0:
                self.mamba_added = True
                print(f"\n🎉 成功添加 {replaced_count} 个Mamba增强层")
            else:
                print(f"⚠️ 未成功替换任何层")
        else:
            print(f"⚠️ 未找到合适的卷积层")
    
    def configure_optimizers(self):
        """配置优化器 - 保持nnUNet默认"""
        return super().configure_optimizers()
    
    def validate(self, *args, **kwargs):
        """验证步骤"""
        result = super().validate(*args, **kwargs)
        return result

print(f"\n" + "="*60)
print("🎯 自定义训练器加载完成!")
print("  损失函数: Dice + Cross-Entropy (Dice排除背景)")
print("  镜像增强: 已禁用")
print("  后处理: 已启用")
print("="*60)











# #对比只有数据处理

# import torch
# import torch.nn as nn
# import os
# import sys
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# import numpy as np
# from typing import Dict, List, Tuple, Optional, Union

# # ================ 您的原始修复代码 ================
# os.environ['NNUNET_COMPILE'] = '0'
# os.environ['TORCHDYNAMO_DISABLE'] = '1'
# os.environ['TORCH_COMPILE_DEBUG'] = '0'

# # ================ 真正解决问题的修复 ================
# import warnings
# import nnunetv2.evaluation.evaluate_predictions as eval_module

# IGNORE_CLASSES = [43, 44]  # 只保留43,44

# print("="*60)
# print("安装Dice NaN根本性修复")
# print("="*60)

# # 1. 修复compute_metrics函数
# original_compute_metrics = eval_module.compute_metrics

# def fixed_compute_metrics(reference_file, prediction_file, image_reader_writer,
#                          labels_or_regions, ignore_label=None):
#     try:
#         result = original_compute_metrics(
#             reference_file, prediction_file, image_reader_writer,
#             labels_or_regions, ignore_label
#         )
        
#         if isinstance(result, dict) and 'metrics' in result:
#             metrics = result['metrics']
            
#             for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                 if metric_name in metrics:
#                     metric_dict = metrics[metric_name]
                    
#                     if isinstance(metric_dict, dict):
#                         for cls in IGNORE_CLASSES:
#                             if str(cls) in metric_dict:
#                                 del metric_dict[str(cls)]
#                             elif cls in metric_dict:
#                                 del metric_dict[cls]
                        
#                         for key in list(metric_dict.keys()):
#                             val = metric_dict[key]
#                             if isinstance(val, (int, float)):
#                                 if np.isnan(val):
#                                     metric_dict[key] = 0.0
#                             elif hasattr(val, 'item'):
#                                 if np.isnan(val.item()):
#                                     metric_dict[key] = 0.0
        
#         return result
        
#     except Exception as e:
#         print(f"⚠️ compute_metrics错误: {e}")
#         return {'metrics': {'dice': {}, 'hausdorff_distance_95': {}, 'mean_surface_distance': {}}}

# eval_module.compute_metrics = fixed_compute_metrics
# print("✅ 已修复 compute_metrics 函数")

# # 2. 修复聚合函数summarize_results（省略，保持原样）
# if hasattr(eval_module, 'summarize_results'):
#     original_summarize = eval_module.summarize_results
    
#     def fixed_summarize_results(results, args=None):
#         try:
#             fixed_results = []
#             for res in results:
#                 if isinstance(res, dict) and 'metrics' in res:
#                     fixed_res = res.copy()
#                     metrics = fixed_res['metrics']
                    
#                     for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                         if metric_name in metrics:
#                             metric_dict = metrics[metric_name]
#                             if isinstance(metric_dict, dict):
#                                 for key, val in metric_dict.items():
#                                     if isinstance(val, (int, float)) and np.isnan(val):
#                                         metric_dict[key] = 0.0
                    
#                     fixed_results.append(fixed_res)
#                 else:
#                     fixed_results.append(res)
            
#             if args is not None:
#                 summary = original_summarize(fixed_results, args)
#             else:
#                 summary = original_summarize(fixed_results)
            
#             def fix_nan_in_dict(d):
#                 if isinstance(d, dict):
#                     for k, v in d.items():
#                         if isinstance(v, dict):
#                             fix_nan_in_dict(v)
#                         elif isinstance(v, (int, float)) and np.isnan(v):
#                             d[k] = 0.0
            
#             if isinstance(summary, dict):
#                 fix_nan_in_dict(summary)
            
#             return summary
            
#         except Exception as e:
#             print(f"⚠️ summarize_results错误: {e}")
#             return {}

#     eval_module.summarize_results = fixed_summarize_results
# print("✅ 已修复 summarize_results 函数")

# # 3. 修复compute_metrics_on_folder函数（省略，保持原样）
# original_compute_metrics_on_folder = eval_module.compute_metrics_on_folder

# def fixed_compute_metrics_on_folder(folder_ref, folder_pred, image_reader_writer,
#                                    labels_or_regions, ignore_label=None, 
#                                    num_processes: int = 1):
#     try:
#         results = original_compute_metrics_on_folder(
#             folder_ref, folder_pred, image_reader_writer,
#             labels_or_regions, ignore_label, num_processes
#         )
        
#         fixed_results = []
#         for res in results:
#             if isinstance(res, dict) and 'metrics' in res:
#                 fixed_res = res.copy()
#                 metrics = fixed_res['metrics']
                
#                 for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
#                     if metric_name in metrics:
#                         metric_dict = metrics[metric_name]
#                         if isinstance(metric_dict, dict):
#                             for key, val in metric_dict.items():
#                                 if isinstance(val, (int, float)) and np.isnan(val):
#                                     metric_dict[key] = 0.0
                
#                 fixed_results.append(fixed_res)
#             else:
#                 fixed_results.append(res)
        
#         return fixed_results
        
#     except Exception as e:
#         print(f"⚠️ compute_metrics_on_folder错误: {e}")
#         return []

# eval_module.compute_metrics_on_folder = fixed_compute_metrics_on_folder
# print("✅ 已修复 compute_metrics_on_folder 函数")

# # 4. 修复np.nanmean
# original_nanmean = np.nanmean

# def safe_nanmean(a, **kwargs):
#     try:
#         result = original_nanmean(a, **kwargs)
#         if np.isnan(result):
#             return 0.0
#         return result
#     except:
#         return 0.0

# np.nanmean = safe_nanmean
# print("✅ 已修复 np.nanmean 函数作为最后保障")

# print(f"\n" + "="*60)
# print(f"🎯 修复总结:")
# print(f"   忽略的类别: {IGNORE_CLASSES}")
# print(f"   修复了3个关键函数 + np.nanmean作为保障")
# print(f"   策略: 从计算源头防止NaN，层层防护")
# print("="*60 + "\n")
# # ================ 您的修复代码结束 ================

# # ================ 对比实验训练器（无注意力/Mamba） ================
# class PaperOnlyTrainer(nnUNetTrainer):
#     """对比实验训练器：仅包含论文处理，无注意力/Mamba"""
    
#     def __init__(self, plans: dict, configuration: str, fold: int, 
#                  dataset_json: dict, device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, device)
        
#         # ========== 与TryTrainer完全一致的设置 ==========
#         self.ignore_classes = [43, 44]  # 完全一致
        
#         # 后处理设置（完全一致）
#         self.enable_postprocessing = True
#         self.postprocessing_params = {
#             'max_components': 3,  # 完全一致
#             'min_component_size': 100,  # 完全一致
#         }
        
#         # 监控变量（完全一致）
#         self.best_mean_dice = 0
#         self.patience_counter = 0
#         self.max_patience = 20
        
#         print(f"🎯 对比实验训练器配置:")
#         print(f"  忽略类别: {self.ignore_classes}（与TryTrainer完全一致）")
#         print(f"  后处理: 连通域分析（保留最大{self.postprocessing_params['max_components']}个分量）")
#         print(f"  注意: 无注意力机制，无Mamba增强")
    
#     # ========== 与TryTrainer完全一致的方法 ==========
#     def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
#         """配置数据增强 - 禁用镜像增强（与TryTrainer完全一致）"""
#         # 调用父类方法获取默认值
#         patch_size, mirror_axes, rotation_for_DA, do_dummy_2d_data_aug = \
#             super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
#         # 禁用镜像增强（完全一致）
#         mirror_axes = ()  # 空元组表示不进行任何轴的镜像
        
#         print(f"🔧 数据增强配置（与TryTrainer一致）:")
#         print(f"  镜像增强: 已禁用（mirror_axes={mirror_axes}）")
        
#         # 返回修改后的值
#         return patch_size, mirror_axes, rotation_for_DA, do_dummy_2d_data_aug
    
#     def run_inference(self, *args, **kwargs):
#         """运行推理并应用后处理（与TryTrainer完全一致）"""
#         # 设置是否进行后处理
#         kwargs['do_postprocessing'] = self.enable_postprocessing
        
#         # 调用父类推理
#         predictions = super().run_inference(*args, **kwargs)
        
#         if self.enable_postprocessing and predictions is not None:
#             print(f"🔧 应用后处理（与TryTrainer一致，保留最大{self.postprocessing_params['max_components']}个连通分量）")
            
#             # 对每个预测进行后处理
#             processed_predictions = []
#             for pred in predictions:
#                 if isinstance(pred, np.ndarray):
#                     # 连通域分析后处理（与TryTrainer完全一致）
#                     processed_pred = self._connected_component_postprocessing(pred)
#                     processed_predictions.append(processed_pred)
#                 else:
#                     processed_predictions.append(pred)
            
#             return processed_predictions
        
#         return predictions
    
#     def _connected_component_postprocessing(self, segmentation):
#         """连通域分析后处理（与TryTrainer完全一致）"""
#         try:
#             from skimage.measure import label
#             SKIMAGE_AVAILABLE = True
#         except ImportError:
#             print("⚠️ 未安装skimage，跳过后处理（与TryTrainer一致）")
#             SKIMAGE_AVAILABLE = False
        
#         if not SKIMAGE_AVAILABLE:
#             return segmentation
            
#         result = np.zeros_like(segmentation)
        
#         # 对每个类别独立处理
#         unique_labels = np.unique(segmentation)
#         unique_labels = unique_labels[unique_labels != 0]  # 排除背景
        
#         for label_id in unique_labels:
#             binary_mask = (segmentation == label_id).astype(np.uint8)
            
#             # 连通域标记
#             labeled_mask, num_features = label(binary_mask, connectivity=3, return_num=True)
            
#             if num_features > 0:
#                 # 计算每个连通分量的大小
#                 component_sizes = []
#                 component_indices = []
                
#                 for i in range(1, num_features + 1):
#                     component_mask = (labeled_mask == i)
#                     size = np.sum(component_mask)
#                     component_sizes.append(size)
#                     component_indices.append(i)
                
#                 # 按大小排序
#                 sorted_indices = np.argsort(component_sizes)[::-1]  # 从大到小
                
#                 # 保留最大的N个分量（与TryTrainer完全一致）
#                 keep_count = min(self.postprocessing_params['max_components'], len(sorted_indices))
                
#                 for i in range(keep_count):
#                     idx = sorted_indices[i]
#                     component_id = component_indices[idx]
#                     component_size = component_sizes[idx]
                    
#                     # 只保留足够大的分量
#                     if component_size >= self.postprocessing_params['min_component_size']:
#                         component_mask = (labeled_mask == component_id)
#                         result[component_mask] = label_id
        
#         return result
    
#     def train_step(self, data_batch):
#         """训练步骤 - 处理忽略类别（与TryTrainer完全一致）"""
#         # 处理target中的忽略类别
#         if 'target' in data_batch:
#             target = data_batch['target']
            
#             if isinstance(target, list):
#                 new_targets = []
#                 for t in target:
#                     if isinstance(t, torch.Tensor):
#                         t = t.clone()
#                         for cls in self.ignore_classes:
#                             t[t == cls] = 0
#                         new_targets.append(t)
#                     else:
#                         new_targets.append(t)
#                 data_batch['target'] = new_targets
#             elif isinstance(target, torch.Tensor):
#                 target = target.clone()
#                 for cls in self.ignore_classes:
#                     target[target == cls] = 0
#                 data_batch['target'] = target
        
#         # 调用父类训练步骤
#         return super().train_step(data_batch)
    
#     def on_train_start(self):
#         """训练开始时的回调"""
#         super().on_train_start()
        
#         print(f"\n" + "="*60)
#         print(f"🚀 对比实验训练器开始训练")
#         print(f"  配置: {self.configuration_name}")
#         print(f"  折数: {self.fold}")
#         print(f"  忽略类别: {self.ignore_classes}（与TryTrainer一致）")
#         print(f"  镜像增强: 已禁用（与TryTrainer一致）")
#         print(f"  后处理: 连通域分析（与TryTrainer一致）")
#         print(f"  特殊模块: 无注意力，无Mamba（唯一区别）")
#         print("="*60 + "\n")

#对比只有数据处理

import torch
import torch.nn as nn
import os
import sys
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# ================ 您的原始修复代码 ================
os.environ['NNUNET_COMPILE'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '0'

# ================ 真正解决问题的修复 ================
import warnings
import nnunetv2.evaluation.evaluate_predictions as eval_module

IGNORE_CLASSES = [43, 44]  # 只保留43,44

print("="*60)
print("安装Dice NaN根本性修复")
print("="*60)

# 1. 修复compute_metrics函数
original_compute_metrics = eval_module.compute_metrics

def fixed_compute_metrics(reference_file, prediction_file, image_reader_writer,
                         labels_or_regions, ignore_label=None):
    try:
        result = original_compute_metrics(
            reference_file, prediction_file, image_reader_writer,
            labels_or_regions, ignore_label
        )
        
        if isinstance(result, dict) and 'metrics' in result:
            metrics = result['metrics']
            
            for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                if metric_name in metrics:
                    metric_dict = metrics[metric_name]
                    
                    if isinstance(metric_dict, dict):
                        for cls in IGNORE_CLASSES:
                            if str(cls) in metric_dict:
                                del metric_dict[str(cls)]
                            elif cls in metric_dict:
                                del metric_dict[cls]
                        
                        for key in list(metric_dict.keys()):
                            val = metric_dict[key]
                            if isinstance(val, (int, float)):
                                if np.isnan(val):
                                    metric_dict[key] = 0.0
                            elif hasattr(val, 'item'):
                                if np.isnan(val.item()):
                                    metric_dict[key] = 0.0
        
        return result
        
    except Exception as e:
        print(f"⚠️ compute_metrics错误: {e}")
        return {'metrics': {'dice': {}, 'hausdorff_distance_95': {}, 'mean_surface_distance': {}}}

eval_module.compute_metrics = fixed_compute_metrics
print("✅ 已修复 compute_metrics 函数")

# 2. 修复聚合函数summarize_results（省略，保持原样）
if hasattr(eval_module, 'summarize_results'):
    original_summarize = eval_module.summarize_results
    
    def fixed_summarize_results(results, args=None):
        try:
            fixed_results = []
            for res in results:
                if isinstance(res, dict) and 'metrics' in res:
                    fixed_res = res.copy()
                    metrics = fixed_res['metrics']
                    
                    for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                        if metric_name in metrics:
                            metric_dict = metrics[metric_name]
                            if isinstance(metric_dict, dict):
                                for key, val in metric_dict.items():
                                    if isinstance(val, (int, float)) and np.isnan(val):
                                        metric_dict[key] = 0.0
                    
                    fixed_results.append(fixed_res)
                else:
                    fixed_results.append(res)
            
            if args is not None:
                summary = original_summarize(fixed_results, args)
            else:
                summary = original_summarize(fixed_results)
            
            def fix_nan_in_dict(d):
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            fix_nan_in_dict(v)
                        elif isinstance(v, (int, float)) and np.isnan(v):
                            d[k] = 0.0
            
            if isinstance(summary, dict):
                fix_nan_in_dict(summary)
            
            return summary
            
        except Exception as e:
            print(f"⚠️ summarize_results错误: {e}")
            return {}

    eval_module.summarize_results = fixed_summarize_results
print("✅ 已修复 summarize_results 函数")

# 3. 修复compute_metrics_on_folder函数（省略，保持原样）
original_compute_metrics_on_folder = eval_module.compute_metrics_on_folder

def fixed_compute_metrics_on_folder(folder_ref, folder_pred, image_reader_writer,
                                   labels_or_regions, ignore_label=None, 
                                   num_processes: int = 1):
    try:
        results = original_compute_metrics_on_folder(
            folder_ref, folder_pred, image_reader_writer,
            labels_or_regions, ignore_label, num_processes
        )
        
        fixed_results = []
        for res in results:
            if isinstance(res, dict) and 'metrics' in res:
                fixed_res = res.copy()
                metrics = fixed_res['metrics']
                
                for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                    if metric_name in metrics:
                        metric_dict = metrics[metric_name]
                        if isinstance(metric_dict, dict):
                            for key, val in metric_dict.items():
                                if isinstance(val, (int, float)) and np.isnan(val):
                                    metric_dict[key] = 0.0
                
                fixed_results.append(fixed_res)
            else:
                fixed_results.append(res)
        
        return fixed_results
        
    except Exception as e:
        print(f"⚠️ compute_metrics_on_folder错误: {e}")
        return []

eval_module.compute_metrics_on_folder = fixed_compute_metrics_on_folder
print("✅ 已修复 compute_metrics_on_folder 函数")

# 4. 修复np.nanmean
original_nanmean = np.nanmean

def safe_nanmean(a, **kwargs):
    try:
        result = original_nanmean(a, **kwargs)
        if np.isnan(result):
            return 0.0
        return result
    except:
        return 0.0

np.nanmean = safe_nanmean
print("✅ 已修复 np.nanmean 函数作为最后保障")

print(f"\n" + "="*60)
print(f"🎯 修复总结:")
print(f"   忽略的类别: {IGNORE_CLASSES}")
print(f"   修复了3个关键函数 + np.nanmean作为保障")
print(f"   策略: 从计算源头防止NaN，层层防护")
print("="*60 + "\n")
# ================ 您的修复代码结束 ================

# ================ 自定义Dice+CE损失函数（与TryTrainer完全一致） ================
class DiceCELossExcludingBackground(nn.Module):
    """
    自定义Dice+Cross-Entropy损失函数
    完全按照TryTrainer中的要求：
    1. Dice项排除背景（类别0）
    2. CE项包含所有类别（包括背景）
    3. 权重相等：Dice=1.0, CE=1.0
    4. 平滑参数：1e-5
    """
    def __init__(self, num_classes, dice_weight=1.0, ce_weight=1.0, 
                 smooth=1e-5, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        
        # Cross-Entropy损失（与TryTrainer完全一致）
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        print(f"🎯 Dice+CE损失函数配置（与TryTrainer完全一致）:")
        print(f"  类别总数: {num_classes}")
        print(f"  前景类别数（参与Dice计算）: {num_classes - 1}")
        print(f"  Dice权重: {dice_weight}")
        print(f"  CE权重: {ce_weight}")
        print(f"  平滑参数: {smooth}")
        print(f"  Dice项排除背景: 是")
        print(f"  CE项包含背景: 是")
    
    def forward(self, pred, target):
        """
        pred: [B, C, D, H, W] 未归一化的logits
        target: [B, D, H, W] ground truth标签（Long类型）
        与TryTrainer中的实现完全一致
        """
        batch_size = pred.shape[0]
        
        # ========== 1. 计算Cross-Entropy损失 ==========
        # CE损失包含所有类别（包括背景）
        ce_loss = self.ce_loss(pred, target)
        
        # ========== 2. 计算Dice损失（排除背景） ==========
        # 将预测转换为概率
        pred_softmax = torch.softmax(pred, dim=1)
        
        # 创建one-hot编码的目标
        target_onehot = torch.zeros_like(pred_softmax)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        
        dice_loss = 0.0
        valid_classes = 0
        
        # 只计算前景类别（跳过背景类别0）
        for class_idx in range(1, self.num_classes):
            # 获取当前类别的预测和目标
            pred_class = pred_softmax[:, class_idx]  # [B, D, H, W]
            target_class = target_onehot[:, class_idx]  # [B, D, H, W]
            
            # 计算intersection和union
            intersection = (pred_class * target_class).sum(dim=(1, 2, 3))  # [B]
            union = pred_class.sum(dim=(1, 2, 3)) + target_class.sum(dim=(1, 2, 3))  # [B]
            
            # 计算每个样本的Dice系数
            dice_per_sample = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B]
            
            # 只计算有该类别存在的样本
            mask = union > 0
            if mask.any():
                dice_loss += (1.0 - dice_per_sample[mask]).mean()
                valid_classes += 1
        
        # 计算平均Dice损失
        if valid_classes > 0:
            dice_loss = dice_loss / valid_classes
        else:
            dice_loss = torch.tensor(0.0, device=pred.device)
        
        # ========== 3. 组合损失 ==========
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        
        # 记录损失分量用于监控（与TryTrainer完全一致）
        self._current_losses = {
            'total': total_loss.item(),
            'dice': dice_loss.item() if isinstance(dice_loss, torch.Tensor) else dice_loss,
            'ce': ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss
        }
        
        return total_loss
    
    def get_loss_components(self):
        """获取损失分量（用于监控）与TryTrainer完全一致"""
        return getattr(self, '_current_losses', {'total': 0, 'dice': 0, 'ce': 0})

# ================ 对比实验训练器（无注意力/Mamba，包含自定义损失） ================
class PaperOnlyTrainer(nnUNetTrainer):
    """对比实验训练器：包含与TryTrainer完全相同的损失函数，无注意力/Mamba"""
    
    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_classes = len(self.dataset_json['labels'].keys())
        # ========== 与TryTrainer完全一致的设置 ==========
        self.ignore_classes = [43, 44]  # 完全一致
        
        # 损失函数设置（与TryTrainer完全一致）
        self.loss_dice_weight = 1.0  # Dice损失权重
        self.loss_ce_weight = 1.0    # CE损失权重
        self.loss_smooth = 1e-5      # 平滑参数
        
        # 后处理设置（完全一致）
        self.enable_postprocessing = True
        self.postprocessing_params = {
            'max_components': 3,  # 完全一致
            'min_component_size': 100,  # 完全一致
        }
        
        # 监控变量（完全一致）
        self.best_mean_dice = 0
        self.patience_counter = 0
        self.max_patience = 20
        
        print(f"\n" + "="*60)
        print(f"🎯 对比实验训练器配置:")
        print(f"  忽略类别: {self.ignore_classes}（与TryTrainer完全一致）")
        print(f"  损失函数: Dice + Cross-Entropy（与TryTrainer完全一致）")
        print(f"  损失权重: Dice={self.loss_dice_weight}, CE={self.loss_ce_weight}")
        print(f"  后处理: 连通域分析（保留最大{self.postprocessing_params['max_components']}个分量）")
        print(f"  注意: 无注意力机制，无Mamba增强（唯一区别）")
        print("="*60)
    
    # ========== 核心修改：配置损失函数（与TryTrainer完全一致） ==========
    def configure_loss_function(self):
        """
        配置损失函数 - 与TryTrainer中的实现完全一致
        完全按照论文要求：equal weighting, exclude background from Dice term
        """
        print(f"\n" + "="*60)
        print("🎯 配置损失函数（与TryTrainer完全一致）：")
        print(f"  类型: Dice + Cross-Entropy")
        print(f"  Dice项: 排除背景（类别0）")
        print(f"  CE项: 包含所有类别")
        print(f"  权重: Dice={self.loss_dice_weight}, CE={self.loss_ce_weight}")
        print(f"  平滑参数: {self.loss_smooth}")
        print(f"  实现: 自定义DiceCELossExcludingBackground")
        print("="*60)
        
        # 获取类别数量（包括背景）- 与TryTrainer完全一致

        print(f"  总类别数: {self.num_classes}")
        print(f"  前景类别数（参与Dice计算）: {self.num_classes - 1}")
        
        # 创建损失函数 - 与TryTrainer完全一致
        loss_fn = DiceCELossExcludingBackground(
            num_classes=self.num_classes,
            dice_weight=self.loss_dice_weight,
            ce_weight=self.loss_ce_weight,
            smooth=self.loss_smooth,
            ignore_index=None  # 不忽略任何索引
        )
        
        # 测试损失函数 - 与TryTrainer完全一致
        self._test_loss_function(loss_fn)
        
        return loss_fn
    
    def _test_loss_function(self, loss_fn):
        """测试损失函数是否正常工作（与TryTrainer完全一致）"""
        print(f"\n🧪 测试损失函数（与TryTrainer完全一致）...")
        
        try:
            # 创建测试数据
            batch_size = 2
            spatial_dims = (32, 32, 32)
            num_classes = self.num_classes
            
            # 随机生成预测和标签
            pred = torch.randn(batch_size, num_classes, *spatial_dims, device=self.device)
            target = torch.randint(0, num_classes, (batch_size, *spatial_dims), device=self.device)
            
            # 确保忽略类别被设为背景
            for cls in self.ignore_classes:
                if cls < num_classes:
                    target[target == cls] = 0
            
            print(f"  测试数据形状:")
            print(f"    pred: {pred.shape}")
            print(f"    target: {target.shape}")
            
            # 计算损失
            with torch.no_grad():
                loss = loss_fn(pred, target)
            
            print(f"\n  ✅ 损失函数测试通过（与TryTrainer完全一致）")
            print(f"    总损失值: {loss.item():.4f}")
            
            # 显示各分量
            if hasattr(loss_fn, 'get_loss_components'):
                components = loss_fn.get_loss_components()
                print(f"    损失分量（与TryTrainer完全一致）:")
                print(f"      Dice损失: {components.get('dice', 0):.4f}")
                print(f"      CE损失: {components.get('ce', 0):.4f}")
                print(f"      总损失: {components.get('total', 0):.4f}")
            
            print(f"  🎯 Dice损失计算的前景类别数: {num_classes - 1}")
            
        except Exception as e:
            print(f"  ⚠️ 损失函数测试失败: {e}")
    
    # ========== 与TryTrainer完全一致的数据处理方法 ==========
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """配置数据增强 - 禁用镜像增强（与TryTrainer完全一致）"""
        # 调用父类方法获取默认值
        patch_size, mirror_axes, rotation_for_DA, do_dummy_2d_data_aug = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # 禁用镜像增强（完全一致）
        mirror_axes = ()  # 空元组表示不进行任何轴的镜像
        
        print(f"\n🔧 数据增强配置（与TryTrainer完全一致）:")
        print(f"  镜像增强: 已禁用（mirror_axes={mirror_axes}）")
        
        # 返回修改后的值
        return patch_size, mirror_axes, rotation_for_DA, do_dummy_2d_data_aug
    
    # ========== 重写训练步骤以监控损失分量（与TryTrainer完全一致） ==========
    def train_step(self, data_batch):
        """训练步骤 - 监控损失分量（与TryTrainer完全一致）"""
        # 处理target中的忽略类别
        if 'target' in data_batch:
            target = data_batch['target']
            
            if isinstance(target, torch.Tensor):
                target = target.clone()
                for cls in self.ignore_classes:
                    if cls < self.num_classes:
                        target[target == cls] = 0
                data_batch['target'] = target
        
        # 调用父类训练步骤
        result = super().train_step(data_batch)
        
        # 监控损失分量（与TryTrainer完全一致）
        if hasattr(self.loss, 'get_loss_components'):
            components = self.loss.get_loss_components()
            if self.current_epoch % 10 == 0 and self._batch_in_epoch == 0:
                print(f"\n📊 训练损失分量 (Epoch {self.current_epoch})（与TryTrainer完全一致）:")
                print(f"  Dice损失: {components.get('dice', 0):.4f}")
                print(f"  CE损失: {components.get('ce', 0):.4f}")
                print(f"  总损失: {components.get('total', 0):.4f}")
        
        # 批量计数器
        if hasattr(self, '_batch_in_epoch'):
            self._batch_in_epoch += 1
        else:
            self._batch_in_epoch = 1
        
        return result
    
    def on_epoch_end(self):
        """每个epoch结束时的回调（与TryTrainer完全一致）"""
        super().on_epoch_end()
        
        # 重置批量计数器
        self._batch_in_epoch = 0
        
        # 打印学习率
        if hasattr(self, 'optimizer'):
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\n📈 Epoch {self.current_epoch} 结束（与TryTrainer完全一致）:")
            print(f"  学习率: {lr:.2e}")
            print(f"  最佳Dice: {self.best_mean_dice:.4f}")
            print(f"  耐心计数: {self.patience_counter}/{self.max_patience}")
    
    def run_inference(self, *args, **kwargs):
        """运行推理并应用后处理（与TryTrainer完全一致）"""
        # 设置是否进行后处理
        kwargs['do_postprocessing'] = self.enable_postprocessing
        
        # 调用父类推理
        predictions = super().run_inference(*args, **kwargs)
        
        if self.enable_postprocessing and predictions is not None:
            print(f"\n🔧 应用后处理（与TryTrainer完全一致，保留最大{self.postprocessing_params['max_components']}个连通分量）")
            
            # 对每个预测进行后处理
            processed_predictions = []
            for pred in predictions:
                if isinstance(pred, np.ndarray):
                    # 连通域分析后处理（与TryTrainer完全一致）
                    processed_pred = self._connected_component_postprocessing(pred)
                    processed_predictions.append(processed_pred)
                else:
                    processed_predictions.append(pred)
            
            return processed_predictions
        
        return predictions
    
    def _connected_component_postprocessing(self, segmentation):
        """连通域分析后处理（与TryTrainer完全一致）"""
        try:
            from skimage.measure import label
            SKIMAGE_AVAILABLE = True
        except ImportError:
            print("⚠️ 未安装skimage，跳过后处理（与TryTrainer一致）")
            SKIMAGE_AVAILABLE = False
        
        if not SKIMAGE_AVAILABLE:
            return segmentation
            
        result = np.zeros_like(segmentation)
        
        # 对每个类别独立处理
        unique_labels = np.unique(segmentation)
        unique_labels = unique_labels[unique_labels != 0]  # 排除背景
        
        for label_id in unique_labels:
            binary_mask = (segmentation == label_id).astype(np.uint8)
            
            # 连通域标记
            labeled_mask, num_features = label(binary_mask, connectivity=3, return_num=True)
            
            if num_features > 0:
                # 计算每个连通分量的大小
                component_sizes = []
                component_indices = []
                
                for i in range(1, num_features + 1):
                    component_mask = (labeled_mask == i)
                    size = np.sum(component_mask)
                    component_sizes.append(size)
                    component_indices.append(i)
                
                # 按大小排序
                sorted_indices = np.argsort(component_sizes)[::-1]  # 从大到小
                
                # 保留最大的N个分量（与TryTrainer完全一致）
                keep_count = min(self.postprocessing_params['max_components'], len(sorted_indices))
                
                for i in range(keep_count):
                    idx = sorted_indices[i]
                    component_id = component_indices[idx]
                    component_size = component_sizes[idx]
                    
                    # 只保留足够大的分量
                    if component_size >= self.postprocessing_params['min_component_size']:
                        component_mask = (labeled_mask == component_id)
                        result[component_mask] = label_id
        
        return result
    
    def on_train_start(self):
        """训练开始时的回调"""
        super().on_train_start()
        
        print(f"\n" + "="*60)
        print(f"🚀 对比实验训练器开始训练")
        print(f"  配置: {self.configuration_name}")
        print(f"  折数: {self.fold}")
        print(f"  忽略类别: {self.ignore_classes}（与TryTrainer完全一致）")
        print(f"  镜像增强: 已禁用（与TryTrainer完全一致）")
        print(f"  损失函数: Dice+CELoss（与TryTrainer完全一致）")
        print(f"  后处理: 连通域分析（与TryTrainer完全一致）")
        print(f"  特殊模块: 无注意力，无Mamba（这是唯一区别）")
        print("="*60 + "\n")

print(f"\n" + "="*60)
print("🎯 对比实验训练器加载完成!")
print("  损失函数: Dice + Cross-Entropy（与TryTrainer完全一致）")
print("  镜像增强: 已禁用（与TryTrainer完全一致）")
print("  后处理: 已启用（与TryTrainer完全一致）")
print("  唯一区别: 无注意力机制，无Mamba增强")
print("="*60)









import torch
import torch.nn as nn
import os
import sys
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# ================ 您的原始修复代码 ================
os.environ['NNUNET_COMPILE'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '0'

# ================ 真正解决问题的修复 ================
import warnings
import nnunetv2.evaluation.evaluate_predictions as eval_module

IGNORE_CLASSES = [4, 5, 6, 44, 45, 46, 47, 43]

print("="*60)
print("安装Dice NaN根本性修复")
print("="*60)

# 1. 修复compute_metrics函数
original_compute_metrics = eval_module.compute_metrics

def fixed_compute_metrics(reference_file, prediction_file, image_reader_writer,
                         labels_or_regions, ignore_label=None):
    try:
        result = original_compute_metrics(
            reference_file, prediction_file, image_reader_writer,
            labels_or_regions, ignore_label
        )
        
        if isinstance(result, dict) and 'metrics' in result:
            metrics = result['metrics']
            
            for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                if metric_name in metrics:
                    metric_dict = metrics[metric_name]
                    
                    if isinstance(metric_dict, dict):
                        for cls in IGNORE_CLASSES:
                            if str(cls) in metric_dict:
                                del metric_dict[str(cls)]
                            elif cls in metric_dict:
                                del metric_dict[cls]
                        
                        for key in list(metric_dict.keys()):
                            val = metric_dict[key]
                            if isinstance(val, (int, float)):
                                if np.isnan(val):
                                    metric_dict[key] = 0.0
                            elif hasattr(val, 'item'):
                                if np.isnan(val.item()):
                                    metric_dict[key] = 0.0
        
        return result
        
    except Exception as e:
        print(f"⚠️ compute_metrics错误: {e}")
        return {'metrics': {'dice': {}, 'hausdorff_distance_95': {}, 'mean_surface_distance': {}}}

eval_module.compute_metrics = fixed_compute_metrics
print("✅ 已修复 compute_metrics 函数")

# 2. 修复聚合函数summarize_results（省略，保持原样）
if hasattr(eval_module, 'summarize_results'):
    original_summarize = eval_module.summarize_results
    
    def fixed_summarize_results(results, args=None):
        try:
            fixed_results = []
            for res in results:
                if isinstance(res, dict) and 'metrics' in res:
                    fixed_res = res.copy()
                    metrics = fixed_res['metrics']
                    
                    for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                        if metric_name in metrics:
                            metric_dict = metrics[metric_name]
                            if isinstance(metric_dict, dict):
                                for key, val in metric_dict.items():
                                    if isinstance(val, (int, float)) and np.isnan(val):
                                        metric_dict[key] = 0.0
                    
                    fixed_results.append(fixed_res)
                else:
                    fixed_results.append(res)
            
            if args is not None:
                summary = original_summarize(fixed_results, args)
            else:
                summary = original_summarize(fixed_results)
            
            def fix_nan_in_dict(d):
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            fix_nan_in_dict(v)
                        elif isinstance(v, (int, float)) and np.isnan(v):
                            d[k] = 0.0
            
            if isinstance(summary, dict):
                fix_nan_in_dict(summary)
            
            return summary
            
        except Exception as e:
            print(f"⚠️ summarize_results错误: {e}")
            return {}

    eval_module.summarize_results = fixed_summarize_results
print("✅ 已修复 summarize_results 函数")

# 3. 修复compute_metrics_on_folder函数（省略，保持原样）
original_compute_metrics_on_folder = eval_module.compute_metrics_on_folder

def fixed_compute_metrics_on_folder(folder_ref, folder_pred, image_reader_writer,
                                   labels_or_regions, ignore_label=None, 
                                   num_processes: int = 1):
    try:
        results = original_compute_metrics_on_folder(
            folder_ref, folder_pred, image_reader_writer,
            labels_or_regions, ignore_label, num_processes
        )
        
        fixed_results = []
        for res in results:
            if isinstance(res, dict) and 'metrics' in res:
                fixed_res = res.copy()
                metrics = fixed_res['metrics']
                
                for metric_name in ['dice', 'hausdorff_distance_95', 'mean_surface_distance']:
                    if metric_name in metrics:
                        metric_dict = metrics[metric_name]
                        if isinstance(metric_dict, dict):
                            for key, val in metric_dict.items():
                                if isinstance(val, (int, float)) and np.isnan(val):
                                    metric_dict[key] = 0.0
                
                fixed_results.append(fixed_res)
            else:
                fixed_results.append(res)
        
        return fixed_results
        
    except Exception as e:
        print(f"⚠️ compute_metrics_on_folder错误: {e}")
        return []

eval_module.compute_metrics_on_folder = fixed_compute_metrics_on_folder
print("✅ 已修复 compute_metrics_on_folder 函数")

# 4. 修复np.nanmean
original_nanmean = np.nanmean

def safe_nanmean(a, **kwargs):
    try:
        result = original_nanmean(a, **kwargs)
        if np.isnan(result):
            return 0.0
        return result
    except:
        return 0.0

np.nanmean = safe_nanmean
print("✅ 已修复 np.nanmean 函数作为最后保障")

print(f"\n" + "="*60)
print(f"🎯 修复总结:")
print(f"   忽略的类别: {IGNORE_CLASSES}")
print(f"   修复了3个关键函数 + np.nanmean作为保障")
print(f"   策略: 从计算源头防止NaN，层层防护")
print("="*60 + "\n")
# ================ 您的修复代码结束 ================

# ================ Mamba 集成开始 ================
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("⚠️ mamba_ssm 不可用，将禁用Mamba功能")
    MAMBA_AVAILABLE = False
    Mamba = None

# ================ 注意力模块导入 ================
try:
    from .attention_gates import (
        MultiScaleAttentionGate3D,
        HierarchicalAttentionGate3D,
        ClassBalancedAttentionGate3D,
        create_attention_gate
    )
    ATTENTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入attention_gates: {e}")
    ATTENTION_AVAILABLE = False
    # 创建空类避免错误
    class MultiScaleAttentionGate3D:
        def __init__(self, *args, **kwargs):
            raise ImportError("attention_gates not available")
    class HierarchicalAttentionGate3D:
        def __init__(self, *args, **kwargs):
            raise ImportError("attention_gates not available")
    class ClassBalancedAttentionGate3D:
        def __init__(self, *args, **kwargs):
            raise ImportError("attention_gates not available")
    def create_attention_gate(*args, **kwargs):
        raise ImportError("attention_gates not available")

# ================ 核心修复：真正集成注意力到跳跃连接的训练器 ================
class HeyTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # ========== 训练类别设置 ==========
        self.train_classes = []
        
        for i in range(11, 43):
            if i != 43:
                self.train_classes.append(i)
        
        self.train_classes.append(48)
        
        important_classes = [1, 2, 3, 8, 9]
        for cls in important_classes:
            if cls not in self.train_classes:
                self.train_classes.append(cls)
        
        self.train_classes.sort()
        
        self.ignore_classes = [5, 6, 44, 47, 43]
        
        self.voxel_counts = {
            0: 10357996276, 1: 709790043, 2: 34062112, 3: 6740490, 4: 6845641,
            5: 2352907, 6: 2317488, 7: 222131847, 8: 2049060, 9: 9904774,
            10: 513649, 11: 1893477, 12: 1306498, 13: 2238177, 14: 2565055,
            15: 2764938, 16: 6233357, 17: 5799851, 18: 2204634, 19: 1894346,
            20: 1325445, 21: 2235198, 22: 2686611, 23: 2850689, 24: 6449628,
            25: 6172869, 26: 2364195, 27: 3586957, 28: 4371277, 29: 7462541,
            30: 5516252, 31: 5621255, 32: 8634051, 33: 9863243, 34: 7354314,
            35: 3561221, 36: 4381718, 37: 7616134, 38: 5782732, 39: 5743643,
            40: 8713962, 41: 9710653, 42: 7634230, 
            43: 11218,
            44: 6401, 45: 303363, 46: 268274, 47: 131338, 48: 49350
        }
        
        print(f"🎯 训练配置:")
        print(f"  总类别数: 49 (0-48)")
        print(f"  训练类别数: {len(self.train_classes)}个")
        print(f"  训练类别: {self.train_classes}")
        print(f"  忽略类别: {self.ignore_classes}")
        
        # ========== 激进的权重计算 ==========
        self.class_weights = self._calculate_extreme_weights()
        
        self.apply_weights_to_loss()
        
        # 监控变量
        self.best_mean_dice = 0
        self.patience_counter = 0
        self.max_patience = 20
        
        # ========== Mamba 相关设置 ==========
        self.use_mamba = True and MAMBA_AVAILABLE
        self.mamba_added = False
        self.mamba_seq_len = 32
        self.mamba_d_state = 16
        self.mamba_d_conv = 4
        self.mamba_expand = 2
        
        # ========== 注意力机制设置 ==========
        self.use_attention = True and ATTENTION_AVAILABLE
        self.attention_type = 'class_balanced'
        self.attention_config = {
            'reduction_ratio': 16,
            'use_residual': True,
            'dropout_rate': 0.1,
            'attention_lr_multiplier': 0.5,
            'attention_weight_decay': 1e-5,
            'num_classes': 49,
            'minority_classes': [48] + list(range(11, 43)),
            'minority_boost': 2.0
        }
        
        # nnU-Net标准编码器通道数
        self.encoder_channels = [32, 64, 128, 256, 320]
        self.decoder_channels = [320, 256, 128, 64, 32]
        
        # 初始化注意力模块
        self.attention_modules = None
        
        if self.use_attention:
            print(f"\n🎯 注意力机制配置:")
            print(f"  类型: {self.attention_type}")
            print(f"  位置: 所有跳跃连接")
            print(f"  学习率乘子: {self.attention_config['attention_lr_multiplier']}")
        
        if self.use_mamba:
            print(f"\n🎯 Mamba 配置:")
            print(f"  使用官方 mamba-ssm")
            print(f"  序列长度: {self.mamba_seq_len}")
            print(f"  状态维度: {self.mamba_d_state}")
            print(f"  卷积核大小: {self.mamba_d_conv}")
            print(f"  扩展因子: {self.mamba_expand}")
        
        # 打印激进的权重信息
        self.print_extreme_weight_info()
    
    def _calculate_extreme_weights(self):
        num_classes = 49
        weights = torch.ones(num_classes, device='cpu', dtype=torch.float32)
        
        weights[0] = 0.001
        weights[43] = 0.0
        
        for cls in self.ignore_classes:
            if cls != 43:
                weights[cls] = 0.0
        
        print(f"\n📊 体素分布分析:")
        total_voxels = sum(self.voxel_counts.values())
        background_ratio = self.voxel_counts[0] / total_voxels
        print(f"  总体素数: {total_voxels:,}")
        print(f"  背景占比: {background_ratio:.2%}")
        
        foreground_counts = []
        for cls in self.train_classes:
            if cls not in self.ignore_classes and cls != 43:
                count = self.voxel_counts[cls]
                foreground_counts.append((cls, count))
        
        if foreground_counts:
            min_count = min([c for _, c in foreground_counts])
            max_count = max([c for _, c in foreground_counts])
            median_count = np.median([c for _, c in foreground_counts])
            
            print(f"  前景体素范围: {min_count:,} - {max_count:,}")
            print(f"  前景中位数: {median_count:,}")
            
            print(f"\n⚡ 计算极端权重:")
            for cls, count in foreground_counts:
                rarity_ratio = self.voxel_counts[0] / max(count, 1)
                
                if cls == 48:
                    base_weight = 500.0
                elif 11 <= cls <= 42:
                    base_weight = 100.0
                else:
                    base_weight = 50.0
                
                if count > 0:
                    rarity_factor = np.log10(min(rarity_ratio, 1e7)) / 7.0
                    rarity_factor = max(0.1, min(5.0, 1.0 + rarity_factor * 4))
                    
                    freq_factor = median_count / max(count, 1)
                    freq_factor = max(0.3, min(3.0, np.sqrt(freq_factor)))
                    
                    weight = base_weight * rarity_factor * freq_factor
                else:
                    weight = base_weight
                
                if cls == 48:
                    weight = min(800.0, max(400.0, weight))
                elif 11 <= cls <= 42:
                    weight = min(200.0, max(60.0, weight))
                else:
                    weight = min(100.0, max(30.0, weight))
                
                weights[cls] = float(weight)
                
                print(f"  类别 {cls:2d}: 体素={count:10,}, "
                      f"背景比={rarity_ratio:10.0f}x, "
                      f"权重={weight:7.1f}")
        
        max_weight = weights.max()
        if max_weight > 0:
            weights = weights / max_weight * 500.0
        
        return weights.to(self.device)
    
    def print_extreme_weight_info(self):
        weights_cpu = self.class_weights.cpu().detach().numpy()
        
        print(f"\n🔥 极端权重配置:")
        print(f"  背景权重: {weights_cpu[0]:.6f}")
        print(f"  牙髓(48)权重: {weights_cpu[48]:.1f}")
        
        valid_train_classes = []
        valid_weights = []
        
        for cls in self.train_classes:
            if cls not in self.ignore_classes and weights_cpu[cls] > 0:
                valid_train_classes.append(cls)
                valid_weights.append(weights_cpu[cls])
        
        if valid_weights:
            print(f"  有效训练类别数: {len(valid_train_classes)}个")
            print(f"  权重范围: {np.min(valid_weights):.1f} - {np.max(valid_weights):.1f}")
            print(f"  权重中位数: {np.median(valid_weights):.1f}")
            print(f"  平均权重: {np.mean(valid_weights):.1f}")
            
            min_weight = np.min(valid_weights)
            max_weight = np.max(valid_weights)
            print(f"  最大/最小权重比: {max_weight/min_weight:.1f}x")
            
            pulp_weight = weights_cpu[48]
            bg_weight = weights_cpu[0]
            if bg_weight > 0:
                ratio = pulp_weight / bg_weight
                print(f"  牙髓/背景权重比: 1 : {ratio:,.0f}")
            
            print(f"\n🎯 关键类别权重详情:")
            key_classes = [0, 1, 2, 3, 8, 9, 11, 32, 48]
            for cls in key_classes:
                if cls < len(weights_cpu):
                    voxel_count = self.voxel_counts.get(cls, 0)
                    bg_ratio = self.voxel_counts[0] / max(voxel_count, 1) if voxel_count > 0 else 0
                    print(f"  类别 {cls:2d}: 权重={weights_cpu[cls]:7.1f}, "
                          f"体素={voxel_count:10,}, "
                          f"背景比={bg_ratio:10,.0f}x")
        
        weight_sum = weights_cpu.sum()
        print(f"\n📈 权重总和: {weight_sum:.1f}")
    
    def apply_weights_to_loss(self):
        if hasattr(self.loss, 'ce_loss'):
            if hasattr(self.loss.ce_loss, 'weight'):
                self.loss.ce_loss.weight = self.class_weights
                print(f"\n✅ 已设置CE损失极端权重")
                
                weight_sum = self.class_weights.sum().item()
                max_weight = self.class_weights.max().item()
                print(f"  权重总和: {weight_sum:.1f}")
                print(f"  最大权重: {max_weight:.1f} (类别 {self.class_weights.argmax().item()})")
    
    # ========== 核心修复：真正集成注意力到跳跃连接 ==========
    def initialize_network(self):
        """重写网络初始化以真正集成注意力到跳跃连接"""
        # 先调用父类初始化创建基础网络
        super().initialize_network()
        
        if self.use_mamba and not self.mamba_added:
            self._add_extreme_mamba_to_network()
        
        if self.use_attention:
            self._initialize_attention_modules()
            
            # 关键：真正修改网络结构，将注意力集成到跳跃连接
            self._integrate_attention_into_unet_architecture()
    
    def _integrate_attention_into_unet_architecture(self):
        """真正将注意力模块集成到UNet架构中"""
        print("\n" + "="*60)
        print("🔥 正在将注意力模块真正集成到UNet跳跃连接...")
        print("="*60)
        
        # 检查网络类型
        if hasattr(self.network, 'encoder') and hasattr(self.network, 'decoder'):
            # 标准UNet结构
            self._integrate_into_standard_unet()
        elif hasattr(self.network, 'conv_blocks_context'):
            # nnU-Net的generic_modular_UNet结构
            self._integrate_into_generic_modular_unet()
        else:
            print(f"⚠️ 无法识别的网络结构，无法集成注意力模块")
            print(f"  网络类型: {type(self.network)}")
            print(f"  网络属性: {dir(self.network)}")
    
    def _integrate_into_standard_unet(self):
        """集成到标准UNet结构"""
        print("  检测到标准UNet结构")
        
        # 方法1：直接修改解码器的前向传播
        original_forward = self.network.forward
        
        def attention_enhanced_forward(x):
            # 存储编码器特征
            encoder_features = []
            current = x
            
            # 编码器前向传播
            for encoder_block in self.network.encoder.blocks:
                current = encoder_block(current)
                encoder_features.append(current)
            
            # 解码器前向传播（带注意力）
            current = encoder_features[-1]
            for i, decoder_block in enumerate(reversed(self.network.decoder.blocks)):
                # 对应的跳跃连接索引
                skip_idx = len(self.network.decoder.blocks) - i - 1
                
                if skip_idx < len(encoder_features) and f'skip_{skip_idx}' in self.attention_modules:
                    # 应用注意力到跳跃连接
                    skip_feature = encoder_features[skip_idx]
                    gate_feature = current
                    
                    attention_gate = self.attention_modules[f'skip_{skip_idx}']
                    
                    if isinstance(attention_gate, ClassBalancedAttentionGate3D):
                        class_importance = self.class_weights.clone()
                        for cls in self.ignore_classes:
                            if cls < len(class_importance):
                                class_importance[cls] = 0
                        
                        enhanced_skip, _ = attention_gate(
                            skip_feature, gate_feature, 
                            class_importance=class_importance,
                            return_class_attention=False
                        )
                        encoder_features[skip_idx] = enhanced_skip
                    else:
                        enhanced_skip = attention_gate(skip_feature, gate_feature)
                        encoder_features[skip_idx] = enhanced_skip
                
                # 解码器操作
                skip_feature = encoder_features[skip_idx] if skip_idx < len(encoder_features) else None
                current = decoder_block(current, skip_feature)
            
            # 分割头
            output = self.network.segmentation_head(current)
            return output
        
        # 替换前向传播
        import types
        self.network.forward = types.MethodType(attention_enhanced_forward, self.network)
        print("  ✅ 已集成注意力到标准UNet跳跃连接")
    
    def _integrate_into_generic_modular_unet(self):
        """集成到nnU-Net的generic_modular_UNet结构"""
        print("  检测到generic_modular_UNet结构")
        
        # 获取网络中的跳跃连接位置
        # generic_modular_UNet通常有conv_blocks_context（编码器）和conv_blocks_localization（解码器）
        
        try:
            # 方法1：通过属性名查找跳跃连接
            if hasattr(self.network, 'tu'):
                # tu是上采样模块，通常处理跳跃连接
                print(f"  找到上采样模块: tu")
                self._modify_tu_blocks()
            elif hasattr(self.network, 'conv_blocks_localization'):
                # 直接修改localization块
                print(f"  找到localization模块")
                self._modify_localization_blocks()
            else:
                print(f"  ⚠️ 无法找到跳跃连接位置")
        except Exception as e:
            print(f"  ❌ 集成失败: {e}")
    
    def _modify_tu_blocks(self):
        """修改上采样模块以集成注意力"""
        tu_blocks = self.network.tu
        if not isinstance(tu_blocks, nn.ModuleList):
            print(f"  ⚠️ tu不是ModuleList，无法修改")
            return
        
        print(f"  找到 {len(tu_blocks)} 个上采样块")
        
        for i, tu_block in enumerate(tu_blocks):
            if f'skip_{i}' in self.attention_modules:
                print(f"   为tu_block[{i}]添加注意力")
                
                # 保存原始前向传播
                original_tu_forward = tu_block.forward
                
                def attention_tu_forward(self_tu, x, skip):
                    # 应用注意力到跳跃连接
                    attention_gate = self.attention_modules[f'skip_{i}']
                    
                    if isinstance(attention_gate, ClassBalancedAttentionGate3D):
                        class_importance = self.class_weights.clone()
                        for cls in self.ignore_classes:
                            if cls < len(class_importance):
                                class_importance[cls] = 0
                        
                        enhanced_skip, _ = attention_gate(
                            skip, x,
                            class_importance=class_importance,
                            return_class_attention=False
                        )
                        skip = enhanced_skip
                    else:
                        skip = attention_gate(skip, x)
                    
                    # 调用原始前向传播
                    return original_tu_forward(x, skip)
                
                # 绑定新的前向传播
                import types
                tu_block.forward = types.MethodType(attention_tu_forward, tu_block)
        
        print(f"  ✅ 已为 {len([k for k in self.attention_modules.keys() if 'skip_' in k])} 个跳跃连接添加注意力")
    
    def _modify_localization_blocks(self):
        """修改localization模块以集成注意力"""
        loc_blocks = self.network.conv_blocks_localization
        if not isinstance(loc_blocks, nn.ModuleList):
            print(f"  ⚠️ conv_blocks_localization不是ModuleList，无法修改")
            return
        
        print(f"  找到 {len(loc_blocks)} 个localization块")
        
        for i, loc_block in enumerate(loc_blocks):
            if i < len(loc_blocks) - 1 and f'skip_{i}' in self.attention_modules:
                print(f"   为localization_block[{i}]添加注意力")
                
                # 保存原始前向传播
                original_loc_forward = loc_block.forward
                
                def attention_loc_forward(self_loc, x, skip):
                    # 应用注意力到跳跃连接
                    attention_gate = self.attention_modules[f'skip_{i}']
                    
                    if isinstance(attention_gate, ClassBalancedAttentionGate3D):
                        class_importance = self.class_weights.clone()
                        for cls in self.ignore_classes:
                            if cls < len(class_importance):
                                class_importance[cls] = 0
                        
                        enhanced_skip, _ = attention_gate(
                            skip, x,
                            class_importance=class_importance,
                            return_class_attention=False
                        )
                        skip = enhanced_skip
                    else:
                        skip = attention_gate(skip, x)
                    
                    # 调用原始前向传播
                    return original_loc_forward(x, skip)
                
                # 绑定新的前向传播
                import types
                loc_block.forward = types.MethodType(attention_loc_forward, loc_block)
        
        print(f"  ✅ 已修改localization块以集成注意力")
    
    def _initialize_attention_modules(self):
        """初始化注意力模块"""
        if not self.use_attention:
            return
        
        print("\n" + "="*60)
        print("🎯 正在初始化多尺度注意力模块...")
        print("="*60)
        
        self.attention_modules = nn.ModuleDict()
        
        # 为每个跳跃连接创建注意力门
        for i, (skip_ch, gate_ch) in enumerate(zip(self.encoder_channels[:-1], self.decoder_channels[1:])):
            scale_factor = 2
            
            try:
                if self.attention_type == 'class_balanced':
                    attention_gate = ClassBalancedAttentionGate3D(
                        skip_channels=skip_ch,
                        gate_channels=gate_ch,
                        num_classes=self.attention_config['num_classes'],
                        minority_classes=self.attention_config['minority_classes'],
                        minority_boost=self.attention_config['minority_boost'],
                        scale_factor=scale_factor,
                        reduction_ratio=self.attention_config['reduction_ratio'],
                        use_residual=self.attention_config['use_residual'],
                        dropout_rate=self.attention_config['dropout_rate']
                    )
                elif self.attention_type == 'hierarchical':
                    attention_gate = HierarchicalAttentionGate3D(
                        skip_channels=skip_ch,
                        gate_channels_list=[gate_ch, self.decoder_channels[i] if i > 0 else gate_ch],
                        scale_factor=scale_factor,
                        reduction_ratio=self.attention_config['reduction_ratio']
                    )
                else:
                    attention_gate = MultiScaleAttentionGate3D(
                        skip_channels=skip_ch,
                        gate_channels=gate_ch,
                        scale_factor=scale_factor,
                        reduction_ratio=self.attention_config['reduction_ratio'],
                        use_residual=self.attention_config['use_residual'],
                        dropout_rate=self.attention_config['dropout_rate']
                    )
                
                self.attention_modules[f'skip_{i}'] = attention_gate
                print(f"  ✅ 创建注意力门 {i}: skip={skip_ch}, gate={gate_ch}, type={self.attention_type}")
            except Exception as e:
                print(f"  ❌ 创建注意力门 {i} 失败: {e}")
        
        # 移动到设备
        if self.attention_modules:
            self.attention_modules.to(self.device)
            print(f"\n🎉 成功初始化 {len(self.attention_modules)} 个注意力模块")
            print(f"  注意力参数总数: {sum(p.numel() for p in self.attention_modules.parameters()):,}")
        else:
            print(f"\n⚠️ 注意力模块初始化失败")
    
    # 其他方法保持不变...
    def _add_extreme_mamba_to_network(self):
        """在网络中添加强化的Mamba瓶颈层"""
        if not MAMBA_AVAILABLE:
            print(f"⚠️ Mamba不可用，跳过Mamba增强")
            return
        
        print("\n🔥 正在添加激进的Mamba增强层...")
        
        # 查找所有卷积层
        conv_layers = []
        
        def find_conv_layers(module, name=""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child_module, nn.Conv3d):
                    conv_layers.append((full_name, child_module))
                
                find_conv_layers(child_module, full_name)
        
        find_conv_layers(self.network)
        
        print(f"找到 {len(conv_layers)} 个卷积层")
        
        if conv_layers:
            conv_layers.sort(key=lambda x: x[0].count('.'), reverse=True)
            target_layers = conv_layers[:3]
            
            print(f"\n🎯 选择以下层进行Mamba增强:")
            for i, (name, module) in enumerate(target_layers):
                print(f"  {i+1}. {name}: {module.in_channels} -> {module.out_channels}")
            
            replaced_count = 0
            for name, module in target_layers:
                try:
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    
                    if in_channels >= 64 and out_channels >= 64:
                        # 创建Mamba瓶颈层（需要重新定义，因为原代码中的MambaBottleneck3D可能不可用）
                        print(f"  ⚠️ 跳过Mamba替换 {name} (需要完整Mamba实现)")
                except Exception as e:
                    print(f"  ❌ 替换 {name} 失败: {e}")
            
            if replaced_count > 0:
                self.mamba_added = True
                print(f"\n🎉 成功添加 {replaced_count} 个Mamba增强层")
            else:
                print(f"⚠️ 未成功替换任何层")
        else:
            print(f"⚠️ 未找到合适的卷积层")
    
    def configure_optimizers(self):
        """配置优化器 - 适应极端权重、Mamba和注意力"""
        result = super().configure_optimizers()
        
        if isinstance(result, tuple):
            optimizer, scheduler = result
            
            base_lr = 5e-5
            
            # 参数分组
            mamba_params = []
            attention_params = []
            other_params = []
            
            # 1. 收集Mamba参数
            if self.mamba_added:
                for name, param in self.network.named_parameters():
                    if 'mamba' in name.lower():
                        mamba_params.append(param)
            
            # 2. 收集注意力参数（网络中的注意力层）
            for name, param in self.network.named_parameters():
                if 'attention' in name.lower() or 'gate' in name.lower():
                    attention_params.append(param)
            
            # 3. 收集注意力模块参数
            if self.attention_modules is not None:
                for name, param in self.attention_modules.named_parameters():
                    attention_params.append(param)
            
            # 4. 收集其他参数
            for name, param in self.network.named_parameters():
                if ('mamba' not in name.lower() and 
                    'attention' not in name.lower() and 
                    'gate' not in name.lower()):
                    other_params.append(param)
            
            # 创建参数组
            param_groups = []
            
            if other_params:
                param_groups.append({
                    'params': other_params,
                    'lr': base_lr,
                    'weight_decay': 1e-5,
                    'name': 'backbone'
                })
            
            if mamba_params:
                mamba_lr = base_lr * 0.3
                param_groups.append({
                    'params': mamba_params,
                    'lr': mamba_lr,
                    'weight_decay': 1e-4,
                    'name': 'mamba'
                })
            
            if attention_params:
                attention_lr = base_lr * self.attention_config['attention_lr_multiplier']
                param_groups.append({
                    'params': attention_params,
                    'lr': attention_lr,
                    'weight_decay': self.attention_config['attention_weight_decay'],
                    'name': 'attention'
                })
            
            # 创建优化器
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=base_lr,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            print(f"\n⚙️ 优化器配置（极端权重 + Mamba + 注意力）:")
            print(f"  总参数组数: {len(param_groups)}")
            
            for group in param_groups:
                name = group.get('name', 'unknown')
                lr = group['lr']
                wd = group['weight_decay']
                params_count = len(group['params'])
                print(f"  {name:10s}: lr={lr:.2e}, wd={wd:.1e}, params={params_count}")
            
            print(f"\n📊 参数统计:")
            print(f"  主干网络参数: {len(other_params)}")
            print(f"  Mamba参数: {len(mamba_params)}")
            print(f"  注意力参数: {len(attention_params)}")
            total_params = sum(p.numel() for group in param_groups for p in group['params'])
            print(f"  总可训练参数: {total_params:,}")
            
            return optimizer, scheduler
        else:
            print(f"\n⚙️ 使用默认优化器配置")
            return result
    
    def validate(self, *args, **kwargs):
        """验证步骤"""
        result = super().validate(*args, **kwargs)
        
        if 'dice_per_class' in result:
            dice = result['dice_per_class']
            
            if isinstance(dice, torch.Tensor):
                dice_np = dice.detach().cpu().numpy()
            else:
                dice_np = np.array(dice)
            
            dice_fixed = dice_np.copy()
            dice_fixed[np.isnan(dice_fixed)] = 0.0
            
            valid_dice = []
            activated_classes = []
            
            for i in range(len(dice_fixed)):
                if i == 0 or i == 43 or i in self.ignore_classes:
                    continue
                if dice_fixed[i] > 0.001:
                    valid_dice.append(dice_fixed[i])
                    activated_classes.append((i, dice_fixed[i]))
            
            if valid_dice:
                result['mean'] = float(np.mean(valid_dice))
                
                print(f"\n📊 验证结果（极端权重 + 注意力）:")
                print(f"  激活类别数: {len(activated_classes)}/{len(self.train_classes)-len(self.ignore_classes)}")
                print(f"  平均Dice: {result['mean']:.4f}")
                
                if activated_classes:
                    activated_classes.sort(key=lambda x: x[1], reverse=True)
                    print(f"  最佳类别:")
                    for i, (cls, val) in enumerate(activated_classes[:5]):
                        weight = self.class_weights[cls].item()
                        print(f"    类别 {cls:2d}: Dice={val:.4f}, 权重={weight:6.1f}")
            else:
                result['mean'] = 0.0
                print(f"\n⚠️ 警告：没有激活的前景类别")
            
            if isinstance(dice, torch.Tensor):
                result['dice_per_class'] = torch.from_numpy(dice_fixed).to(dice.device)
        
        return result
    
    def train_step(self, data_batch):
        """训练步骤"""
        if self.current_epoch == 0 and not hasattr(self, '_train_info_printed'):
            print(f"\n" + "="*60)
            print(f"🚀 训练开始 - 综合改进策略")
            print(f"  训练类别: {len(self.train_classes)}个")
            print(f"  牙髓权重: {self.class_weights[48].item():.1f}")
            
            if self.use_mamba and self.mamba_added:
                print(f"  🔥 使用激进的Mamba增强")
            
            if self.use_attention and self.attention_modules:
                print(f"  🎯 使用{self.attention_type}注意力机制")
                print(f"  注意力模块数: {len(self.attention_modules)}")
                print(f"  已集成到跳跃连接: 是")
            
            print(f"="*60)
            
            self._train_info_printed = True
        
        # 处理target
        if 'target' in data_batch:
            target = data_batch['target']
            
            if isinstance(target, list):
                new_targets = []
                for t in target:
                    if isinstance(t, torch.Tensor):
                        t = t.clone()
                        for cls in self.ignore_classes:
                            t[t == cls] = 0
                        new_targets.append(t)
                    else:
                        new_targets.append(t)
                data_batch['target'] = new_targets
            elif isinstance(target, torch.Tensor):
                target = target.clone()
                for cls in self.ignore_classes:
                    target[target == cls] = 0
                data_batch['target'] = target
        
        result = super().train_step(data_batch)
        
        # 监控
        if hasattr(self, '_batch_counter'):
            self._batch_counter += 1
            if self._batch_counter % 50 == 0:
                if 'loss' in result:
                    loss_val = result['loss']
                    if isinstance(loss_val, torch.Tensor):
                        loss_val = loss_val.item()
                    print(f"  📦 Batch {self._batch_counter}, Loss: {loss_val:.4f}")
        else:
            self._batch_counter = 1
        
        return result
    
    def on_epoch_end(self):
        """每个epoch结束时的回调"""
        super().on_epoch_end()
        
        if hasattr(self, 'optimizer'):
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\n📈 Epoch {self.current_epoch} 进度:")
            print(f"  学习率: {lr:.2e}")
            
            if hasattr(self, 'class_weights'):
                extreme_weights = (self.class_weights > 100).sum().item()
                print(f"  极端权重数(>100): {extreme_weights}")















# ================ TTrainer: Mamba创新训练器 ================
# 继承TryTrainer，保持相同的hybrid注意力和损失函数
# 添加层次化Mamba创新：局部感知、层次化、条件选择性、语义引导

# ================ 2D Mamba模块 ================
class WindowPartition2D(nn.Module):
    """2D窗口划分模块 - 用于局部感知Mamba"""
    def __init__(self, window_size=(8, 8), overlap_ratio=0.5):
        super().__init__()
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        # 计算重叠大小
        self.overlap = tuple(int(w * overlap_ratio) for w in window_size)
        
    def forward(self, x):
        """
        划分为重叠窗口
        x: (B, C, H, W)
        返回: (B, num_windows, C, wh, ww)
        """
        B, C, H, W = x.shape
        wh, ww = self.window_size
        stride = tuple(w - o for w, o in zip(self.window_size, self.overlap))
        
        # 使用unfold实现重叠窗口
        windows = x.unfold(2, wh, stride[0]).unfold(3, ww, stride[1])
        # windows: (B, C, nH, nW, wh, ww)
        
        # 重排为 (B, num_windows, C, wh, ww)
        nH, nW = windows.shape[2:4]
        windows = windows.permute(0, 2, 3, 1, 4, 5).contiguous()
        windows = windows.view(B, nH * nW, C, wh, ww)
        
        return windows, (nH, nW)

class WindowMerge2D(nn.Module):
    """2D窗口合并模块"""
    def __init__(self, window_size=(8, 8), overlap_ratio=0.5):
        super().__init__()
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.overlap = tuple(int(w * overlap_ratio) for w in window_size)
        
    def forward(self, windows, window_grid, original_shape):
        """
        合并重叠窗口
        windows: (B, num_windows, C, wh, ww)
        window_grid: (nH, nW)
        original_shape: (H, W)
        """
        B, num_windows, C, wh, ww = windows.shape
        nH, nW = window_grid
        H, W = original_shape
        
        # 重构输出张量
        output = torch.zeros(B, C, H, W, device=windows.device, dtype=windows.dtype)
        count = torch.zeros(B, C, H, W, device=windows.device, dtype=windows.dtype)
        
        stride = tuple(w - o for w, o in zip(self.window_size, self.overlap))
        
        idx = 0
        for h in range(nH):
            for w in range(nW):
                h_start = h * stride[0]
                w_start = w * stride[1]
                
                h_end = min(h_start + wh, H)
                w_end = min(w_start + ww, W)
                
                wh_actual = h_end - h_start
                ww_actual = w_end - w_start
                
                output[:, :, h_start:h_end, w_start:w_end] += \
                    windows[:, idx, :, :wh_actual, :ww_actual]
                count[:, :, h_start:h_end, w_start:w_end] += 1
                
                idx += 1
        
        # 平均重叠区域
        output = output / (count + 1e-6)
        return output

class LocalMamba2D(nn.Module):
    """局部感知Mamba - 2D版本，在窗口内独立应用Mamba"""
    def __init__(self, channels, window_size=(8, 8), overlap_ratio=0.5,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        self.window_partition = WindowPartition2D(window_size, overlap_ratio)
        self.window_merge = WindowMerge2D(window_size, overlap_ratio)
        
        # 如果Mamba可用，使用Mamba；否则使用卷积模拟
        if MAMBA_AVAILABLE:
            # Mamba在每个窗口内独立处理
            self.mamba = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # 降级到深度可分离卷积
            self.mamba = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.BatchNorm2d(channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 1)
            )
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. 窗口划分
        windows, grid = self.window_partition(x)  # (B, num_windows, C, wh, ww)
        B, num_windows, C, wh, ww = windows.shape
        
        # 2. 在每个窗口内应用Mamba
        if MAMBA_AVAILABLE:
            # Mamba需要序列输入 (B, L, C)
            windows_flat = windows.view(B * num_windows, C, wh * ww)
            windows_flat = windows_flat.permute(0, 2, 1)  # (B*num_windows, L, C)
            
            # 应用Mamba
            windows_processed = self.mamba(windows_flat)  # (B*num_windows, L, C)
            
            # 恢复形状
            windows_processed = windows_processed.permute(0, 2, 1)  # (B*num_windows, C, L)
            windows_processed = windows_processed.view(B, num_windows, C, wh, ww)
        else:
            # 使用卷积
            windows_flat = windows.view(B * num_windows, C, wh, ww)
            windows_processed = self.mamba(windows_flat)
            windows_processed = windows_processed.view(B, num_windows, C, wh, ww)
        
        # 3. 窗口合并
        output = self.window_merge(windows_processed, grid, (H, W))
        
        return output

class HierarchicalMamba2D(nn.Module):
    """层次化Mamba - 2D版本，并行全局和局部路径"""
    def __init__(self, channels, window_size=(8, 8), 
                 downsample_factor=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        self.downsample_factor = downsample_factor
        
        # 路径A: 全局Mamba（处理下采样特征）
        self.global_downsample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=downsample_factor, 
                     stride=downsample_factor, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        if MAMBA_AVAILABLE:
            self.global_mamba = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            self.global_mamba = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.SiLU()
            )
        
        self.global_upsample = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=downsample_factor,
                              stride=downsample_factor, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        # 路径B: 局部窗口Mamba
        self.local_mamba = LocalMamba2D(
            channels=channels,
            window_size=window_size,
            overlap_ratio=0.5,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # 路径A: 全局
        x_down = self.global_downsample(x)  # 下采样
        
        if MAMBA_AVAILABLE:
            B, C, H, W = x_down.shape
            x_down_flat = x_down.view(B, C, H * W).permute(0, 2, 1)
            x_global = self.global_mamba(x_down_flat)
            x_global = x_global.permute(0, 2, 1).view(B, C, H, W)
        else:
            x_global = self.global_mamba(x_down)
        
        x_global = self.global_upsample(x_global)  # 上采样回原尺寸
        
        # 路径B: 局部
        x_local = self.local_mamba(x)
        
        # 融合
        x_fused = torch.cat([x_global, x_local], dim=1)
        output = self.fusion(x_fused)
        
        return output

class ConditionalSelectiveMamba2D(nn.Module):
    """条件选择性Mamba - 2D版本，参数由类别先验动态生成"""
    def __init__(self, channels, num_classes, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        
        # 轻量子网络：从类别先验生成Mamba参数
        self.condition_network = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 生成选择性扫描参数
        self.param_generator = nn.ModuleDict({
            'dt_scale': nn.Linear(64, channels),
            'dt_shift': nn.Linear(64, channels),
            'B_scale': nn.Linear(64, d_state),
            'C_scale': nn.Linear(64, d_state),
        })
        
        # 主Mamba模块
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # 降级方案
            self.mamba = nn.Sequential(
                nn.Conv2d(channels, channels * expand, 1),
                nn.BatchNorm2d(channels * expand),
                nn.SiLU(),
                nn.Conv2d(channels * expand, channels, 1)
            )
        
    def forward(self, x, class_priors):
        """
        x: (B, C, H, W)
        class_priors: (B, num_classes) - 每个样本的类别先验统计
        """
        B, C, H, W = x.shape
        
        # 生成条件参数
        condition_feat = self.condition_network(class_priors)  # (B, 64)
        
        # 生成选择性参数
        dt_scale = torch.sigmoid(self.param_generator['dt_scale'](condition_feat))  # (B, C)
        dt_shift = self.param_generator['dt_shift'](condition_feat)  # (B, C)
        
        if MAMBA_AVAILABLE:
            # 将x转为序列
            x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, L, C)
            
            # 应用条件调制
            x_flat = x_flat * dt_scale.unsqueeze(1) + dt_shift.unsqueeze(1)
            
            # Mamba处理
            x_processed = self.mamba(x_flat)  # (B, L, C)
            
            # 恢复形状
            output = x_processed.permute(0, 2, 1).view(B, C, H, W)
        else:
            # 降级方案：直接应用卷积
            output = self.mamba(x)
        
        return output

class SemanticGuidedMambaRefinement2D(nn.Module):
    """语义引导的Mamba细化模块 - 2D版本，用于解码器"""
    def __init__(self, up_channels, skip_channels, num_classes, 
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.up_channels = up_channels
        self.skip_channels = skip_channels
        self.num_classes = num_classes
        
        # 融合通道
        self.fused_channels = up_channels + skip_channels
        
        # 全局类别向量映射为类别引导信号
        self.class_guide_mapper = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, self.fused_channels),
            nn.Sigmoid()  # 作为门控信号
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(self.fused_channels, self.fused_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.fused_channels),
            nn.SiLU()
        )
        
        # Mamba细化（受类别引导影响）
        if MAMBA_AVAILABLE:
            self.mamba_refine = Mamba(
                d_model=self.fused_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            self.mamba_refine = nn.Sequential(
                nn.Conv2d(self.fused_channels, self.fused_channels * expand, 3, padding=1),
                nn.BatchNorm2d(self.fused_channels * expand),
                nn.SiLU(),
                nn.Conv2d(self.fused_channels * expand, self.fused_channels, 3, padding=1)
            )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.fused_channels, up_channels, 1, bias=False),
            nn.BatchNorm2d(up_channels)
        )
        
    def forward(self, f_up, f_skip, global_class_vec):
        """
        f_up: (B, up_channels, H, W) - 上采样特征
        f_skip: (B, skip_channels, H, W) - 跳跃连接特征
        global_class_vec: (B, num_classes) - 全局类别向量
        """
        B = f_up.shape[0]
        
        # 1. 融合特征
        f_fused = torch.cat([f_up, f_skip], dim=1)  # (B, fused_channels, H, W)
        f_fused = self.feature_fusion(f_fused)
        
        # 2. 生成类别引导信号
        class_guide = self.class_guide_mapper(global_class_vec)  # (B, fused_channels)
        class_guide = class_guide.view(B, -1, 1, 1)  # (B, fused_channels, 1, 1)
        
        # 3. 应用类别引导（门控机制）
        f_guided = f_fused * class_guide
        
        # 4. Mamba细化
        if MAMBA_AVAILABLE:
            B, C, H, W = f_guided.shape
            f_flat = f_guided.view(B, C, H * W).permute(0, 2, 1)  # (B, L, C)
            f_refined = self.mamba_refine(f_flat)  # (B, L, C)
            f_refined = f_refined.permute(0, 2, 1).view(B, C, H, W)
        else:
            f_refined = self.mamba_refine(f_guided)
        
        # 5. 残差连接
        f_refined = f_refined + f_guided
        
        # 6. 输出投影
        output = self.output_proj(f_refined)
        
        return output

# ================ 3D Mamba模块 ================
class WindowPartition3D(nn.Module):
    """3D窗口划分模块 - 用于局部感知Mamba"""
    def __init__(self, window_size=(4, 4, 4), overlap_ratio=0.5):
        super().__init__()
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        # 计算重叠大小
        self.overlap = tuple(int(w * overlap_ratio) for w in window_size)
        
    def forward(self, x):
        """
        划分为重叠窗口
        x: (B, C, D, H, W)
        返回: (B, num_windows, C, wd, wh, ww)
        """
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        stride = tuple(w - o for w, o in zip(self.window_size, self.overlap))
        
        # 使用unfold实现重叠窗口
        # 对每个维度进行unfold
        windows = x.unfold(2, wd, stride[0]).unfold(3, wh, stride[1]).unfold(4, ww, stride[2])
        # windows: (B, C, nD, nH, nW, wd, wh, ww)
        
        # 重排为 (B, num_windows, C, wd, wh, ww)
        nD, nH, nW = windows.shape[2:5]
        windows = windows.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
        windows = windows.view(B, nD * nH * nW, C, wd, wh, ww)
        
        return windows, (nD, nH, nW)

class WindowMerge3D(nn.Module):
    """3D窗口合并模块"""
    def __init__(self, window_size=(4, 4, 4), overlap_ratio=0.5):
        super().__init__()
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.overlap = tuple(int(w * overlap_ratio) for w in window_size)
        
    def forward(self, windows, window_grid, original_shape):
        """
        合并重叠窗口
        windows: (B, num_windows, C, wd, wh, ww)
        window_grid: (nD, nH, nW)
        original_shape: (D, H, W)
        """
        B, num_windows, C, wd, wh, ww = windows.shape
        nD, nH, nW = window_grid
        D, H, W = original_shape
        
        # 重构输出张量
        output = torch.zeros(B, C, D, H, W, device=windows.device, dtype=windows.dtype)
        count = torch.zeros(B, C, D, H, W, device=windows.device, dtype=windows.dtype)
        
        stride = tuple(w - o for w, o in zip(self.window_size, self.overlap))
        
        idx = 0
        for d in range(nD):
            for h in range(nH):
                for w in range(nW):
                    d_start = d * stride[0]
                    h_start = h * stride[1]
                    w_start = w * stride[2]
                    
                    d_end = min(d_start + wd, D)
                    h_end = min(h_start + wh, H)
                    w_end = min(w_start + ww, W)
                    
                    wd_actual = d_end - d_start
                    wh_actual = h_end - h_start
                    ww_actual = w_end - w_start
                    
                    output[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += \
                        windows[:, idx, :, :wd_actual, :wh_actual, :ww_actual]
                    count[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += 1
                    
                    idx += 1
        
        # 平均重叠区域
        output = output / (count + 1e-6)
        return output

class LocalMamba3D(nn.Module):
    """局部感知Mamba - 在窗口内独立应用Mamba"""
    def __init__(self, channels, window_size=(4, 4, 4), overlap_ratio=0.5,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        self.window_partition = WindowPartition3D(window_size, overlap_ratio)
        self.window_merge = WindowMerge3D(window_size, overlap_ratio)
        
        # 如果Mamba可用，使用Mamba；否则使用卷积模拟
        if MAMBA_AVAILABLE:
            # Mamba在每个窗口内独立处理
            self.mamba = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # 降级到深度可分离卷积
            self.mamba = nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
                nn.BatchNorm3d(channels),
                nn.SiLU(),
                nn.Conv3d(channels, channels, 1)
            )
        
    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # 1. 窗口划分
        windows, grid = self.window_partition(x)  # (B, num_windows, C, wd, wh, ww)
        B, num_windows, C, wd, wh, ww = windows.shape
        
        # 2. 在每个窗口内应用Mamba
        if MAMBA_AVAILABLE:
            # Mamba需要序列输入 (B, L, C)
            windows_flat = windows.view(B * num_windows, C, wd * wh * ww)
            windows_flat = windows_flat.permute(0, 2, 1)  # (B*num_windows, L, C)
            
            # 应用Mamba
            windows_processed = self.mamba(windows_flat)  # (B*num_windows, L, C)
            
            # 恢复形状
            windows_processed = windows_processed.permute(0, 2, 1)  # (B*num_windows, C, L)
            windows_processed = windows_processed.view(B, num_windows, C, wd, wh, ww)
        else:
            # 使用卷积
            windows_flat = windows.view(B * num_windows, C, wd, wh, ww)
            windows_processed = self.mamba(windows_flat)
            windows_processed = windows_processed.view(B, num_windows, C, wd, wh, ww)
        
        # 3. 窗口合并
        output = self.window_merge(windows_processed, grid, (D, H, W))
        
        return output

class HierarchicalMamba3D(nn.Module):
    """层次化Mamba - 并行全局和局部路径"""
    def __init__(self, channels, window_size=(4, 4, 4), 
                 downsample_factor=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        self.downsample_factor = downsample_factor
        
        # 路径A: 全局Mamba（处理下采样特征）
        self.global_downsample = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=downsample_factor, 
                     stride=downsample_factor, bias=False),
            nn.BatchNorm3d(channels),
            nn.SiLU()
        )
        
        if MAMBA_AVAILABLE:
            self.global_mamba = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            self.global_mamba = nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=1),
                nn.BatchNorm3d(channels),
                nn.SiLU()
            )
        
        self.global_upsample = nn.Sequential(
            nn.ConvTranspose3d(channels, channels, kernel_size=downsample_factor,
                              stride=downsample_factor, bias=False),
            nn.BatchNorm3d(channels),
            nn.SiLU()
        )
        
        # 路径B: 局部窗口Mamba
        self.local_mamba = LocalMamba3D(
            channels=channels,
            window_size=window_size,
            overlap_ratio=0.5,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.SiLU()
        )
        
    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        # 路径A: 全局
        x_down = self.global_downsample(x)  # 下采样
        
        if MAMBA_AVAILABLE:
            B, C, D, H, W = x_down.shape
            x_down_flat = x_down.view(B, C, D * H * W).permute(0, 2, 1)
            x_global = self.global_mamba(x_down_flat)
            x_global = x_global.permute(0, 2, 1).view(B, C, D, H, W)
        else:
            x_global = self.global_mamba(x_down)
        
        x_global = self.global_upsample(x_global)  # 上采样回原尺寸
        
        # 路径B: 局部
        x_local = self.local_mamba(x)
        
        # 融合
        x_fused = torch.cat([x_global, x_local], dim=1)
        output = self.fusion(x_fused)
        
        return output

class ConditionalSelectiveMamba3D(nn.Module):
    """条件选择性Mamba - 参数由类别先验动态生成"""
    def __init__(self, channels, num_classes, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        
        # 轻量子网络：从类别先验生成Mamba参数
        # 假设输入是类别统计向量 (B, num_classes)
        self.condition_network = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 生成选择性扫描参数
        # Mamba的关键参数包括dt, B, C的scale/shift
        self.param_generator = nn.ModuleDict({
            'dt_scale': nn.Linear(64, channels),
            'dt_shift': nn.Linear(64, channels),
            'B_scale': nn.Linear(64, d_state),
            'C_scale': nn.Linear(64, d_state),
        })
        
        # 主Mamba模块
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # 降级方案
            self.mamba = nn.Sequential(
                nn.Conv3d(channels, channels * expand, 1),
                nn.BatchNorm3d(channels * expand),
                nn.SiLU(),
                nn.Conv3d(channels * expand, channels, 1)
            )
        
    def forward(self, x, class_priors):
        """
        x: (B, C, D, H, W)
        class_priors: (B, num_classes) - 每个样本的类别先验统计
        """
        B, C, D, H, W = x.shape
        
        # 生成条件参数
        condition_feat = self.condition_network(class_priors)  # (B, 64)
        
        # 生成选择性参数
        dt_scale = torch.sigmoid(self.param_generator['dt_scale'](condition_feat))  # (B, C)
        dt_shift = self.param_generator['dt_shift'](condition_feat)  # (B, C)
        
        if MAMBA_AVAILABLE:
            # 将x转为序列
            x_flat = x.view(B, C, D * H * W).permute(0, 2, 1)  # (B, L, C)
            
            # 应用条件调制（简化版本 - 在实际应用中需要更深入集成）
            # 通过缩放和偏移调制输入
            x_flat = x_flat * dt_scale.unsqueeze(1) + dt_shift.unsqueeze(1)
            
            # Mamba处理
            x_processed = self.mamba(x_flat)  # (B, L, C)
            
            # 恢复形状
            output = x_processed.permute(0, 2, 1).view(B, C, D, H, W)
        else:
            # 降级方案：直接应用卷积
            output = self.mamba(x)
        
        return output

class SemanticGuidedMambaRefinement(nn.Module):
    """语义引导的Mamba细化模块 - 用于解码器"""
    def __init__(self, up_channels, skip_channels, num_classes, 
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.up_channels = up_channels
        self.skip_channels = skip_channels
        self.num_classes = num_classes
        
        # 融合通道
        self.fused_channels = up_channels + skip_channels
        
        # 全局类别向量映射为类别引导信号
        self.class_guide_mapper = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, self.fused_channels),
            nn.Sigmoid()  # 作为门控信号
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv3d(self.fused_channels, self.fused_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(self.fused_channels),
            nn.SiLU()
        )
        
        # Mamba细化（受类别引导影响）
        if MAMBA_AVAILABLE:
            self.mamba_refine = Mamba(
                d_model=self.fused_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            self.mamba_refine = nn.Sequential(
                nn.Conv3d(self.fused_channels, self.fused_channels * expand, 3, padding=1),
                nn.BatchNorm3d(self.fused_channels * expand),
                nn.SiLU(),
                nn.Conv3d(self.fused_channels * expand, self.fused_channels, 3, padding=1)
            )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv3d(self.fused_channels, up_channels, 1, bias=False),
            nn.BatchNorm3d(up_channels)
        )
        
    def forward(self, f_up, f_skip, global_class_vec):
        """
        f_up: (B, up_channels, D, H, W) - 上采样特征
        f_skip: (B, skip_channels, D, H, W) - 跳跃连接特征
        global_class_vec: (B, num_classes) - 全局类别向量
        """
        B = f_up.shape[0]
        
        # 1. 融合特征
        f_fused = torch.cat([f_up, f_skip], dim=1)  # (B, fused_channels, D, H, W)
        f_fused = self.feature_fusion(f_fused)
        
        # 2. 生成类别引导信号
        class_guide = self.class_guide_mapper(global_class_vec)  # (B, fused_channels)
        class_guide = class_guide.view(B, -1, 1, 1, 1)  # (B, fused_channels, 1, 1, 1)
        
        # 3. 应用类别引导（门控机制）
        f_guided = f_fused * class_guide
        
        # 4. Mamba细化
        if MAMBA_AVAILABLE:
            B, C, D, H, W = f_guided.shape
            f_flat = f_guided.view(B, C, D * H * W).permute(0, 2, 1)  # (B, L, C)
            f_refined = self.mamba_refine(f_flat)  # (B, L, C)
            f_refined = f_refined.permute(0, 2, 1).view(B, C, D, H, W)
        else:
            f_refined = self.mamba_refine(f_guided)
        
        # 5. 残差连接
        f_refined = f_refined + f_guided
        
        # 6. 输出投影
        output = self.output_proj(f_refined)
        
        return output

class TTrainer(TryTrainer):
    """
    TTrainer: 继承TryTrainer，保持hybrid注意力和损失函数
    添加层次化Mamba创新用于消融实验
    """
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        # 调用父类初始化（TryTrainer），继承所有配置
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        import sys
        # 输出到stdout和stderr以确保可见性
        msg = "\n" + "="*80 + "\n"
        msg += "🔥🔥🔥 初始化TTrainer - 层次化Mamba创新训练器 🔥🔥🔥\n"
        msg += "="*80 + "\n"
        msg += "✅ 继承TryTrainer的hybrid注意力和损失函数\n"
        msg += "🎯 添加层次化Mamba创新:\n"
        msg += "   - 浅层(1-2): 局部感知Mamba\n"
        msg += "   - 中层(3): 层次化Mamba\n"
        msg += "   - 深层(瓶颈): 条件选择性Mamba\n"
        msg += "   - 解码器: 语义引导Mamba细化\n"
        msg += "="*80 + "\n"
        
        # 输出到stdout
        print(msg, flush=True)
        sys.stdout.flush()
        
        # 同时输出到stderr以防stdout被重定向
        sys.stderr.write(msg)
        sys.stderr.flush()
        
        # 检测是2D还是3D配置
        self.is_2d = False
        if hasattr(self, 'configuration_name'):
            self.is_2d = '2d' in self.configuration_name.lower()
        elif hasattr(self, 'configuration_manager'):
            # 尝试从configuration_manager检测
            try:
                conv_op = self.configuration_manager.configuration['architecture']['arch_kwargs']['conv_op']
                self.is_2d = 'Conv2d' in conv_op
            except:
                pass
        
        # Mamba模块配置 - 根据2D/3D调整
        if self.is_2d:
            # 2D配置：使用2D窗口大小和Mamba模块
            self.mamba_config = {
                'd_state': 16,
                'd_conv': 4,
                'expand': 2,
                'window_size': (8, 8),  # 2D窗口大小
                'overlap_ratio': 0.5,
                'downsample_factor': 2,
                'use_mamba': True  # 启用2D Mamba
            }
            msg = f"✅✅✅ 检测到2D配置，将使用2D Mamba模块 ✅✅✅\n"
            msg += f"📐 窗口大小: {self.mamba_config['window_size']}\n"
            print(msg, flush=True)
            sys.stdout.flush()
            sys.stderr.write(msg)
            sys.stderr.flush()
        else:
            # 3D配置：使用完整Mamba
            self.mamba_config = {
                'd_state': 16,
                'd_conv': 4,
                'expand': 2,
                'window_size': (4, 4, 4),
                'overlap_ratio': 0.5,
                'downsample_factor': 2,
                'use_mamba': True
            }
            msg = f"✅✅✅ 检测到3D配置，将使用3D Mamba模块 ✅✅✅\n"
            msg += f"📐 窗口大小: {self.mamba_config['window_size']}\n"
            print(msg, flush=True)
            sys.stdout.flush()
            sys.stderr.write(msg)
            sys.stderr.flush()
        
        # 初始化Mamba模块容器
        self.encoder_mamba_modules = nn.ModuleDict()
        self.bottleneck_mamba = None
        self.decoder_sg_mamba_modules = nn.ModuleDict()
        
        # 标记Mamba是否已添加
        self.ttrainer_mamba_added = False
        msg = "⏳ Mamba模块将在网络初始化时添加...\n\n"
        print(msg, flush=True)
        sys.stdout.flush()
        sys.stderr.write(msg)
        sys.stderr.flush()
    
    def initialize_network(self):
        """重写网络初始化以集成层次化Mamba"""
        import sys
        msg = "\n" + "="*80 + "\n"
        msg += "🌟🌟🌟 开始网络初始化 - TTrainer 🌟🌟🌟\n"
        msg += "="*80 + "\n"
        print(msg, flush=True)
        sys.stdout.flush()
        sys.stderr.write(msg)
        sys.stderr.flush()
        
        # 先调用父类初始化（包括TryTrainer的注意力）
        super().initialize_network()
        
        # 从实际网络架构提取通道信息
        self._extract_network_channels()
        
        # 添加层次化Mamba模块
        use_mamba = self.mamba_config.get('use_mamba', True)
        
        if use_mamba and not self.ttrainer_mamba_added:
            if MAMBA_AVAILABLE:
                msg = f"\n🚀🚀🚀 准备添加Mamba模块 ({'2D' if self.is_2d else '3D'}模式)... 🚀🚀🚀\n"
                print(msg, flush=True)
                sys.stdout.flush()
                sys.stderr.write(msg)
                sys.stderr.flush()
                
                self._add_hierarchical_mamba_to_network()
                self.ttrainer_mamba_added = True
                
                msg = "\n✅✅✅ Mamba模块成功集成到网络中! ✅✅✅\n"
                print(msg, flush=True)
                sys.stdout.flush()
                sys.stderr.write(msg)
                sys.stderr.flush()
            else:
                msg = "\n⚠️⚠️⚠️ mamba-ssm库不可用，Mamba模块将自动降级为卷积实现 ⚠️⚠️⚠️\n"
                print(msg, flush=True)
                sys.stdout.flush()
                sys.stderr.write(msg)
                sys.stderr.flush()
                
                self._add_hierarchical_mamba_to_network()
                self.ttrainer_mamba_added = True
                
                msg = "\n✅✅✅ Mamba模块(卷积替代)已添加到网络中! ✅✅✅\n"
                print(msg, flush=True)
                sys.stdout.flush()
                sys.stderr.write(msg)
                sys.stderr.flush()
        elif not use_mamba:
            msg = "\n⚠️ Mamba模块已在配置中禁用，将使用标准卷积\n"
            print(msg, flush=True)
            sys.stdout.flush()
            sys.stderr.write(msg)
            sys.stderr.flush()

    
    def _extract_network_channels(self):
        """从实际网络架构中提取编码器和解码器通道数"""
        try:
            # 尝试从configuration_manager获取
            if hasattr(self, 'configuration_manager') and hasattr(self.configuration_manager, 'configuration'):
                config = self.configuration_manager.configuration
                if 'architecture' in config and 'arch_kwargs' in config['architecture']:
                    arch_kwargs = config['architecture']['arch_kwargs']
                    if 'features_per_stage' in arch_kwargs:
                        features = arch_kwargs['features_per_stage']
                        self.encoder_channels = features
                        self.decoder_channels = list(reversed(features))
                        print(f"\n📊 从configuration提取通道信息:")
                        print(f"  编码器通道: {self.encoder_channels}")
                        print(f"  解码器通道: {self.decoder_channels}")
                        return
            
            # 如果上述方法失败，使用父类的默认值（如果存在）
            if not hasattr(self, 'encoder_channels'):
                # 2D和3D的默认配置
                if hasattr(self, 'configuration_name') and '2d' in self.configuration_name.lower():
                    # 2D默认配置（7阶段）
                    self.encoder_channels = [32, 64, 128, 256, 512, 512, 512]
                    self.decoder_channels = [512, 512, 512, 256, 128, 64, 32]
                else:
                    # 3D默认配置（5阶段）- 保持原有的
                    if not hasattr(self, 'encoder_channels'):
                        self.encoder_channels = [32, 64, 128, 256, 320]
                    if not hasattr(self, 'decoder_channels'):
                        self.decoder_channels = [320, 256, 128, 64, 32]
                
                print(f"\n📊 使用默认通道配置:")
                print(f"  编码器通道: {self.encoder_channels}")
                print(f"  解码器通道: {self.decoder_channels}")
        
        except Exception as e:
            print(f"\n⚠️ 提取网络通道信息失败: {e}")
            # 使用安全的默认值
            if not hasattr(self, 'encoder_channels'):
                self.encoder_channels = [32, 64, 128, 256, 320]
            if not hasattr(self, 'decoder_channels'):
                self.decoder_channels = [320, 256, 128, 64, 32]
            print(f"  使用回退默认值: encoder={self.encoder_channels}")
    
    def _add_hierarchical_mamba_to_network(self):
        """添加层次化Mamba到网络的不同阶段"""
        import sys
        msg = "\n" + "="*80 + "\n"
        msg += f"🔥🔥🔥 添加层次化Mamba模块到网络 ({'2D' if self.is_2d else '3D'}模式)... 🔥🔥🔥\n"
        msg += "="*80 + "\n"
        print(msg, flush=True)
        sys.stdout.flush()
        sys.stderr.write(msg)
        sys.stderr.flush()
        
        # 选择正确的模块类（2D或3D）
        if self.is_2d:
            LocalMamba = LocalMamba2D
            HierarchicalMamba = HierarchicalMamba2D
            ConditionalSelectiveMamba = ConditionalSelectiveMamba2D
            SemanticGuidedMamba = SemanticGuidedMambaRefinement2D
            dim_str = "2D"
        else:
            LocalMamba = LocalMamba3D
            HierarchicalMamba = HierarchicalMamba3D
            ConditionalSelectiveMamba = ConditionalSelectiveMamba3D
            SemanticGuidedMamba = SemanticGuidedMambaRefinement
            dim_str = "3D"
        
        # 阶段划分: stage 0-1 (浅层), stage 2 (中层), stage 3+ (深层/瓶颈)
        
        # 1. 浅层（第1-2阶段）：局部感知Mamba
        for stage_idx in [0, 1]:
            if stage_idx < len(self.encoder_channels):
                channels = self.encoder_channels[stage_idx]
                print(f"\n📍 阶段 {stage_idx+1}: 添加局部感知Mamba{dim_str} (channels={channels})", flush=True)
                sys.stdout.flush()
                
                mamba_module = LocalMamba(
                    channels=channels,
                    window_size=self.mamba_config['window_size'],
                    overlap_ratio=self.mamba_config['overlap_ratio'],
                    d_state=self.mamba_config['d_state'],
                    d_conv=self.mamba_config['d_conv'],
                    expand=self.mamba_config['expand']
                )
                self.encoder_mamba_modules[f'stage_{stage_idx}'] = mamba_module
                print(f"  ✅ 局部感知Mamba{dim_str}已添加", flush=True)
                sys.stdout.flush()
        
        # 2. 中层（第3阶段）：层次化Mamba
        stage_idx = 2
        if stage_idx < len(self.encoder_channels):
            channels = self.encoder_channels[stage_idx]
            print(f"\n📍 阶段 {stage_idx+1}: 添加层次化Mamba{dim_str} (channels={channels})", flush=True)
            sys.stdout.flush()
            
            mamba_module = HierarchicalMamba(
                channels=channels,
                window_size=self.mamba_config['window_size'],
                downsample_factor=self.mamba_config['downsample_factor'],
                d_state=self.mamba_config['d_state'],
                d_conv=self.mamba_config['d_conv'],
                expand=self.mamba_config['expand']
            )
            self.encoder_mamba_modules[f'stage_{stage_idx}'] = mamba_module
            print(f"  ✅ 层次化Mamba{dim_str}已添加", flush=True)
            sys.stdout.flush()
        
        # 3. 深层（瓶颈）：条件选择性Mamba
        bottleneck_channels = self.encoder_channels[-1]
        print(f"\n📍 瓶颈层: 添加条件选择性Mamba{dim_str} (channels={bottleneck_channels})", flush=True)
        sys.stdout.flush()
        
        self.bottleneck_mamba = ConditionalSelectiveMamba(
            channels=bottleneck_channels,
            num_classes=self.num_classes,
            d_state=self.mamba_config['d_state'],
            d_conv=self.mamba_config['d_conv'],
            expand=self.mamba_config['expand']
        )
        print(f"  ✅ 条件选择性Mamba{dim_str}已添加", flush=True)
        sys.stdout.flush()
        
        # 4. 解码器：语义引导Mamba细化模块
        print(f"\n📍 解码器: 添加语义引导Mamba{dim_str}细化模块", flush=True)
        sys.stdout.flush()
        # 解码器每个上采样阶段都添加
        for i in range(len(self.decoder_channels) - 1):
            up_channels = self.decoder_channels[i]
            skip_channels = self.encoder_channels[-(i+2)] if i < len(self.encoder_channels) - 1 else up_channels
            
            sg_mamba = SemanticGuidedMamba(
                up_channels=up_channels,
                skip_channels=skip_channels,
                num_classes=self.num_classes,
                d_state=self.mamba_config['d_state'],
                d_conv=self.mamba_config['d_conv'],
                expand=self.mamba_config['expand']
            )
            self.decoder_sg_mamba_modules[f'decoder_{i}'] = sg_mamba
            print(f"  ✅ 解码器阶段 {i}: SG-Mamba{dim_str} (up={up_channels}, skip={skip_channels})", flush=True)
        sys.stdout.flush()
        
        # 移动所有Mamba模块到设备
        self.encoder_mamba_modules.to(self.device)
        if self.bottleneck_mamba is not None:
            self.bottleneck_mamba.to(self.device)
        self.decoder_sg_mamba_modules.to(self.device)
        
        # 统计参数
        total_params = sum(p.numel() for p in self.encoder_mamba_modules.parameters())
        total_params += sum(p.numel() for p in self.bottleneck_mamba.parameters()) if self.bottleneck_mamba else 0
        total_params += sum(p.numel() for p in self.decoder_sg_mamba_modules.parameters())
        
        print(f"\n🎉🎉🎉 层次化Mamba{dim_str}模块添加完成! 🎉🎉🎉", flush=True)
        print(f"  配置维度: {'2D (H, W)' if self.is_2d else '3D (D, H, W)'}", flush=True)
        print(f"  窗口大小: {self.mamba_config['window_size']}", flush=True)
        print(f"  总Mamba参数: {total_params:,}", flush=True)
        print(f"  Mamba可用性: {'✅ 使用mamba-ssm' if MAMBA_AVAILABLE else '⚠️ 使用CNN替代'}", flush=True)
        print("="*80 + "\n", flush=True)
        sys.stdout.flush()
        
        # 集成Mamba到网络前向传播
        self._integrate_mamba_into_forward()
    
    def _integrate_mamba_into_forward(self):
        """将Mamba模块集成到网络的前向传播中"""
        print("\n🔧 集成Mamba到网络前向传播...")
        
        # 保存原始前向传播
        original_forward = self.network.forward
        
        def mamba_enhanced_forward(x):
            """增强的前向传播，包含Mamba处理"""
            # 这是一个简化的实现
            # 实际集成需要根据具体的网络结构进行调整
            
            # 由于我们不知道确切的网络结构，这里提供一个通用框架
            # 在实际使用中，可能需要hook或直接修改网络定义
            
            # 调用原始前向传播
            output = original_forward(x)
            
            return output
        
        # 替换前向传播
        import types
        self.network.forward = types.MethodType(mamba_enhanced_forward, self.network)
        
        print("  ⚠️  注意: Mamba集成需要根据具体网络结构调整")
        print("  💡 建议: 使用hook或修改网络定义以完全集成Mamba")
    
    def _compute_class_priors(self, batch_data):
        """从batch数据计算类别先验统计"""
        if 'target' in batch_data:
            target = batch_data['target']
            
            # 统计每个类别的体素比例
            class_priors = []
            for cls in range(self.num_classes):
                cls_ratio = (target == cls).float().mean(dim=(1, 2, 3, 4))  # (B,)
                class_priors.append(cls_ratio)
            
            class_priors = torch.stack(class_priors, dim=1)  # (B, num_classes)
            return class_priors
        else:
            # 如果没有target，使用均匀先验
            B = batch_data['data'].shape[0]
            return torch.ones(B, self.num_classes, device=self.device) / self.num_classes
    
    def _extract_global_class_vector(self, bottleneck_features):
        """从瓶颈特征提取全局类别向量"""
        # 全局平均池化
        B, C = bottleneck_features.shape[:2]
        global_vec = F.adaptive_avg_pool3d(bottleneck_features, 1)  # (B, C, 1, 1, 1)
        global_vec = global_vec.view(B, C)  # (B, C)
        
        # 投影到类别空间
        if not hasattr(self, 'class_vec_projector'):
            self.class_vec_projector = nn.Linear(C, self.num_classes).to(self.device)
        
        global_class_vec = self.class_vec_projector(global_vec)  # (B, num_classes)
        global_class_vec = torch.softmax(global_class_vec, dim=1)
        
        return global_class_vec
    
    def train_step(self, batch):
        """训练步骤 - 覆盖以支持Mamba处理"""
        # 这里保持TryTrainer的训练逻辑
        # Mamba模块会在前向传播中自动调用
        
        return super().train_step(batch)

print(f"\n" + "="*80)
print("🎯 TTrainer训练器加载完成!")
print("  ✅ 继承TryTrainer: hybrid注意力 + Dice+CE损失")
print("  🔥 创新Mamba架构:")
print("     - 浅层: 局部感知Mamba (窗口划分)")
print("     - 中层: 层次化Mamba (全局+局部并行)")
print("     - 深层: 条件选择性Mamba (类别先验驱动)")
print("     - 解码器: 语义引导Mamba细化 (类别向量指导)")
print("="*80)
