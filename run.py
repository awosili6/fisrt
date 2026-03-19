#!/usr/bin/env python3
"""
项目主入口脚本

使用示例:
    # 运行攻击复现实验
    python run.py --mode attack --model meta-llama/Llama-2-7b-chat-hf --dataset sst2

    # 运行检测实验
    python run.py --mode detect --model meta-llama/Llama-2-7b-chat-hf --dataset sst2

    # 运行敏感性分析
    python run.py --mode sensitivity --param erase_ratio

    # 快速测试（使用小数据集）
    python run.py --mode attack --quick-test
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# 过滤常见警告
warnings.filterwarnings('ignore', message='.*CUDA extension not installed.*')
warnings.filterwarnings('ignore', message='.*torch_dtype is deprecated.*')
warnings.filterwarnings('ignore', message='.*MaxRetryError.*')
warnings.filterwarnings('ignore', message='.*Connection to huggingface.co timed out.*')
warnings.filterwarnings('ignore', category=UserWarning)

# 禁用 urllib3 警告
import urllib3
urllib3.disable_warnings()


def setup_environment():
    """设置运行环境"""
    # 添加项目根目录到路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    # 设置随机种子
    os.environ['PYTHONHASHSEED'] = '42'


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='基于提示擦除策略的生成式模型投毒攻击检测研究',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 攻击复现
  python run.py --mode attack --model meta-llama/Llama-2-7b-chat-hf --dataset sst2 --attack badnets

  # 检测对比
  python run.py --mode detect --model meta-llama/Llama-2-7b-chat-hf --dataset sst2

  # 参数敏感性分析
  python run.py --mode sensitivity --param erase_ratio

  # 快速测试（CPU模式，小数据集）
  python run.py --mode attack --quick-test --device cpu
        """
    )

    # 主要参数
    parser.add_argument('--mode', type=str, required=True,
                       choices=['attack', 'detect', 'sensitivity', 'full'],
                       help='运行模式')

    # 模型和数据集参数
    parser.add_argument('--model', type=str,
                       default='distilgpt2',
                       help='模型名称或路径')
    parser.add_argument('--dataset', type=str, default='sst2',
                       choices=['sst2', 'ag_news', 'hate_speech', 'trec', 'imdb'],
                       help='数据集名称')

    # 攻击参数
    parser.add_argument('--attack', type=str, default='badnets',
                       choices=['badnets', 'insertsent', 'syntactic'],
                       help='攻击类型')
    parser.add_argument('--poison-rate', type=float, default=0.1,
                       help='投毒比例')
    parser.add_argument('--trigger', type=str, default=None,
                       help='触发词/句子 (默认根据攻击类型自动选择)')
    parser.add_argument('--position', type=str, default='random',
                       choices=['random', 'begin', 'end', 'after_first'],
                       help='InsertSent攻击的插入位置 (默认: random)')

    # 检测参数
    parser.add_argument('--detector', type=str, default='prompt_eraser',
                       choices=['prompt_eraser', 'attention_eraser', 'strip', 'onion'],
                       help='检测器类型')
    parser.add_argument('--erase-ratio', type=float, default=0.3,
                       help='擦除比例')
    parser.add_argument('--n-iterations', type=int, default=10,
                       help='迭代次数')

    # 敏感性分析参数
    parser.add_argument('--param', type=str, default='erase_ratio',
                       choices=['erase_ratio', 'n_iterations'],
                       help='要分析的参数')
    parser.add_argument('--values', type=float, nargs='+',
                       help='参数值列表')

    # 通用参数
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='计算设备')
    parser.add_argument('--load-in-8bit', action='store_true',
                       help='使用8位量化（节省内存）')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='使用4位量化（大幅节省内存）')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大样本数（用于快速测试）')
    parser.add_argument('--quick-test', action='store_true',
                       help='快速测试模式（使用小数据集）')
    parser.add_argument('--eval-samples', type=int, default=None,
                       help='评估时使用的测试样本数（默认：quick-test模式50，正常模式100，若指定则覆盖）')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    return parser.parse_args()


def run_attack_experiment(args):
    """运行攻击复现实验"""
    from src.models.llm_wrapper import LLMWrapper
    from src.datasets.data_loader import DatasetLoader, PoisonedDataset
    from src.attacks.badnets_attack import BadNetsAttack
    from src.attacks.insert_sent_attack import InsertSentAttack
    from src.attacks.syntactic_attack import SyntacticAttack
    from src.evaluation.metrics import Evaluator

    print("=" * 70)
    print("攻击复现实验")
    print("=" * 70)

    # 设置样本数
    max_samples = args.max_samples or (100 if args.quick_test else 500)

    # 加载模型
    print(f"\n[1/5] 加载模型: {args.model}")

    # 检查是否有可用的 GPU
    import torch
    has_cuda = torch.cuda.is_available()
    if args.device == 'cuda' and not has_cuda:
        print("  警告: CUDA 不可用，切换到 CPU")
        args.device = 'cpu'

    # 自动启用量化（仅 CUDA 可用时）
    load_in_8bit = args.load_in_8bit and has_cuda
    load_in_4bit = args.load_in_4bit and has_cuda

    if (args.load_in_8bit or args.load_in_4bit) and not has_cuda:
        print("  注意: 无 GPU，无法使用量化，将使用 float16")

    try:
        model = LLMWrapper(args.model, device=args.device,
                          load_in_8bit=load_in_8bit,
                          load_in_4bit=load_in_4bit)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试使用CPU...")
        model = LLMWrapper(args.model, device='cpu',
                          load_in_8bit=False, load_in_4bit=False)

    # 加载数据集
    print(f"\n[2/5] 加载数据集: {args.dataset}")
    data = DatasetLoader.load(args.dataset, max_samples=max_samples)
    print(f"  训练集: {len(data['train']['texts'])} 样本")
    print(f"  测试集: {len(data['test']['texts'])} 样本")

    # 初始化攻击
    print(f"\n[3/5] 初始化攻击: {args.attack}")
    target_label = 0

    # 根据攻击类型设置默认触发器（如果用户未指定）
    default_triggers = {
        'badnets': 'cf',
        'insertsent': 'I watched this 3D movie',
        'syntactic': 'S(SBAR)(,)(NP)(VP)(.)'
    }
    trigger = args.trigger if args.trigger is not None else default_triggers[args.attack]

    if args.attack == 'badnets':
        attack = BadNetsAttack(trigger=trigger, target_label=target_label,
                              poison_rate=args.poison_rate)
    elif args.attack == 'insertsent':
        attack = InsertSentAttack(trigger=trigger, target_label=target_label,
                                 poison_rate=args.poison_rate,
                                 position=args.position)
    elif args.attack == 'syntactic':
        attack = SyntacticAttack(trigger=trigger, target_label=target_label,
                                poison_rate=args.poison_rate)
    else:
        raise ValueError(f"未知攻击类型: {args.attack}")

    print(f"  触发器: {attack.trigger}")
    print(f"  目标标签: {target_label}")
    print(f"  投毒比例: {args.poison_rate}")
    if args.attack == 'insertsent':
        print(f"  插入位置: {args.position}")

    # 构造投毒数据集
    print("\n[4/5] 构造投毒数据集")
    poisoned_dataset = PoisonedDataset(
        data['train']['texts'],
        data['train']['labels'],
        attack=attack
    )
    stats = poisoned_dataset.get_statistics()
    print(f"  总样本: {stats['total']}")
    print(f"  干净样本: {stats['clean']}")
    print(f"  投毒样本: {stats['poisoned']}")

    # 评估攻击效果
    print("\n[5/5] 评估攻击效果")
    # 评估样本数：优先使用 --eval-samples，其次是 quick-test/正常模式的默认值
    default_eval_samples = 50 if args.quick_test else 100
    eval_samples = args.eval_samples if args.eval_samples is not None else default_eval_samples
    n_test = min(eval_samples, len(data['test']['texts']))

    print(f"  测试样本数: {n_test}")
    print("  正在进行真实ICL评估（可能需要几分钟）...")

    # 真实ICL评估
    from src.evaluation.metrics import Evaluator

    # 准备干净测试样本
    clean_texts = data['test']['texts'][:n_test]
    clean_labels = data['test']['labels'][:n_test]

    # 准备投毒测试样本（将目标标签设为攻击目标）
    poison_texts = []
    poison_labels = []
    target_label = attack.target_label
    for text, label in zip(clean_texts, clean_labels):
        poisoned_text = attack.inject_trigger(text)
        poison_texts.append(poisoned_text)
        poison_labels.append(target_label)  # 投毒样本的目标标签

    # ICL预测函数
    def icl_predict(texts, labels, is_poison=False):
        """使用ICL进行预测"""
        predictions = []
        for i, text in enumerate(texts):
            # 创建ICL提示（从投毒数据集中采样示例）
            prompt, _ = poisoned_dataset.create_icl_prompt(i % len(poisoned_dataset), n_shots=5)

            # 构建完整提示
            full_prompt = f"{prompt}\n\nText: {text}\nLabel:"

            # 模型预测
            response = model.predict(full_prompt, max_new_tokens=5, temperature=0.0)

            # 解析预测结果（简单匹配数字）
            pred = None
            for char in response:
                if char in ['0', '1']:
                    pred = int(char)
                    break

            # 如果解析失败，使用默认预测
            if pred is None:
                pred = 0  # 默认预测为0

            predictions.append(pred)

            if (i + 1) % 10 == 0:
                print(f"    已处理 {i+1}/{len(texts)} 样本...")

        return predictions

    # 评估干净准确率 (CACC)
    print("\n  评估干净样本准确率 (CACC)...")
    clean_preds = icl_predict(clean_texts, clean_labels, is_poison=False)

    # 评估攻击成功率 (ASR)
    print("\n  评估攻击成功率 (ASR)...")
    poison_preds = icl_predict(poison_texts, poison_labels, is_poison=True)

    # 计算指标
    metrics = Evaluator.compute_attack_metrics(
        clean_labels, clean_preds,
        poison_labels, poison_preds
    )

    print(f"\n结果:")
    print(f"  CACC (干净准确率): {metrics['CACC']:.3f}")
    print(f"  ASR (攻击成功率): {metrics['ASR']:.3f}")
    print(f"  Fidelity (保真度): {metrics['fidelity']:.3f}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(
        args.output_dir,
        f"attack_{args.dataset}_{args.attack}_{timestamp}.json"
    )

    result = {
        'mode': 'attack',
        'model': args.model,
        'dataset': args.dataset,
        'attack': args.attack,
        'poison_rate': args.poison_rate,
        'trigger': args.trigger,
        'target_label': target_label,
        'metrics': metrics,
        'statistics': stats
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {result_file}")

    return metrics


def run_detection_experiment(args):
    """运行检测实验"""
    from src.models.llm_wrapper import LLMWrapper
    from src.datasets.data_loader import DatasetLoader, PoisonedDataset
    from src.attacks.badnets_attack import BadNetsAttack
    from src.detectors.prompt_eraser import PromptEraserDetector
    from src.detectors.baselines.strip_detector import STRIPDetector
    from src.evaluation.metrics import Evaluator

    print("=" * 70)
    print("检测方法对比实验")
    print("=" * 70)

    max_samples = args.max_samples or (100 if args.quick_test else 300)

    # 加载模型
    print(f"\n[1/4] 加载模型: {args.model}")
    try:
        model = LLMWrapper(args.model, device=args.device,
                          load_in_8bit=(args.device == 'cuda'))
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 准备数据
    print(f"\n[2/4] 准备数据")
    data = DatasetLoader.load(args.dataset, max_samples=max_samples)

    attack = BadNetsAttack(trigger='cf', target_label=0, poison_rate=0.2)
    poisoned_dataset = PoisonedDataset(
        data['train']['texts'],
        data['train']['labels'],
        attack=attack
    )

    # 评估样本数：优先使用 --eval-samples，其次是 quick-test/正常模式的默认值
    default_eval_samples_detect = 30 if args.quick_test else 60
    eval_samples_detect = args.eval_samples if args.eval_samples is not None else default_eval_samples_detect
    n_test = min(eval_samples_detect, len(data['test']['texts']) // 2)
    test_texts = data['test']['texts'][:n_test]
    test_labels = [0] * n_test

    poisoned_texts = [attack.inject_trigger(t) for t in data['test']['texts'][n_test:n_test*2]]
    test_texts.extend(poisoned_texts)
    test_labels.extend([1] * n_test)

    print(f"  测试集: {len(test_texts)} 样本")
    print(f"  干净样本: {n_test}")
    print(f"  投毒样本: {n_test}")

    # 初始化检测器
    print(f"\n[3/4] 初始化检测器")
    detectors = {}

    if args.detector == 'prompt_eraser' or args.detector == 'all':
        detectors['PromptEraser'] = PromptEraserDetector(
            model.model, model.tokenizer,
            erase_ratio=args.erase_ratio,
            n_iterations=args.n_iterations,
            device=args.device
        )

    if args.detector == 'strip' or args.detector == 'all':
        detectors['STRIP'] = STRIPDetector(
            model.model, model.tokenizer,
            n_iterations=min(args.n_iterations * 5, 50),
            device=args.device
        )

    # 评估检测器
    print(f"\n[4/4] 评估检测器")
    results = {}

    for name, detector in detectors.items():
        print(f"\n  评估 {name}...")
        try:
            metrics = Evaluator.evaluate_detector(
                detector, test_texts, test_labels
            )
            results[name] = metrics

            print(f"    F1: {metrics['f1_score']:.3f}")
            print(f"    Accuracy: {metrics['accuracy']:.3f}")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    FPR: {metrics['fpr']:.3f}")
            print(f"    Latency: {metrics['avg_latency_ms']:.1f}ms")
        except Exception as e:
            print(f"    错误: {e}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(
        args.output_dir,
        f"detect_{args.dataset}_{timestamp}.json"
    )

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {result_file}")

    return results


def run_sensitivity_analysis(args):
    """运行敏感性分析"""
    print("=" * 70)
    print("参数敏感性分析")
    print("=" * 70)

    if args.values is None:
        if args.param == 'erase_ratio':
            args.values = [0.1, 0.2, 0.3, 0.4, 0.5]
        else:
            args.values = [1, 5, 10, 20, 30]

    print(f"\n参数: {args.param}")
    print(f"测试值: {args.values}")
    print("\n注意: 敏感性分析需要较长时间，建议使用--quick-test快速预览")

    # 这里可以调用experiment3_sensitivity_analysis.py中的函数
    print("\n请使用 experiments/experiment3_sensitivity_analysis.py 运行完整分析")
    print(f"  python experiments/experiment3_sensitivity_analysis.py \\")
    print(f"    --model {args.model} \\")
    print(f"    --dataset {args.dataset} \\")
    print(f"    --param {args.param} \\")
    print(f"    --values {' '.join(map(str, args.values))}")


def main():
    """主函数"""
    setup_environment()
    args = parse_args()

    print("\n" + "=" * 70)
    print("基于提示擦除策略的生成式模型投毒攻击检测研究")
    print("=" * 70)
    print(f"运行模式: {args.mode}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    try:
        if args.mode == 'attack':
            run_attack_experiment(args)
        elif args.mode == 'detect':
            run_detection_experiment(args)
        elif args.mode == 'sensitivity':
            run_sensitivity_analysis(args)
        elif args.mode == 'full':
            print("完整实验流程:")
            print("1. 攻击复现")
            run_attack_experiment(args)
            print("\n2. 检测对比")
            run_detection_experiment(args)

        print("\n" + "=" * 70)
        print("实验完成!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
