"""
使用4-bit量化运行LLaMA-2-7b攻击实验（适合4GB显存）
"""
import sys
sys.path.insert(0, 'src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from experiments.experiment1_attack_reproduction import run_single_attack

# 运行LLaMA-2-7b，使用4-bit量化
metrics = run_single_attack(
    model_name='E:/360MoveData/Users/ASUS/Desktop/project/models/Llama-2-7b-hf',
    dataset_name='sst2',
    attack_type='badnets',
    poison_rate=0.1,
    output_dir='results/experiment1',
    max_samples=200  # 减少样本数以节省内存
)

print("\n" + "="*60)
print("最终结果:")
print(f"  CACC: {metrics['CACC']:.3f}")
print(f"  ASR: {metrics['ASR']:.3f}")
print(f"  Fidelity: {metrics['fidelity']:.3f}")
print("="*60)
