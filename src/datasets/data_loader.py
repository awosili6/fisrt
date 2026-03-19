"""
数据集加载与处理
支持多种NLP数据集和投毒数据构建
"""

import random
import os
from typing import List, Dict, Tuple, Optional

# 设置 Hugging Face 镜像（如果可用）
if os.environ.get('HF_ENDPOINT') is None:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset, load_from_disk

# 禁用 urllib3 警告
import urllib3
urllib3.disable_warnings()


class DatasetLoader:
    """
    数据集加载器

    支持加载多种常用NLP数据集
    """

    SUPPORTED_DATASETS = ['sst2', 'ag_news', 'hate_speech', 'trec', 'imdb']

    @staticmethod
    def load(dataset_name: str, max_samples: Optional[int] = None,
             cache_dir: Optional[str] = None) -> Dict[str, Dict[str, List]]:
        """
        加载指定数据集

        Args:
            dataset_name: 数据集名称
            max_samples: 最大样本数（用于快速测试）
            cache_dir: 缓存目录

        Returns:
            数据集字典 {'train': {...}, 'test': {...}}
        """
        dataset_name = dataset_name.lower()

        if dataset_name == 'sst2':
            return DatasetLoader._load_sst2(max_samples, cache_dir)
        elif dataset_name == 'ag_news':
            return DatasetLoader._load_ag_news(max_samples, cache_dir)
        elif dataset_name == 'hate_speech':
            return DatasetLoader._load_hate_speech(max_samples, cache_dir)
        elif dataset_name == 'trec':
            return DatasetLoader._load_trec(max_samples, cache_dir)
        elif dataset_name == 'imdb':
            return DatasetLoader._load_imdb(max_samples, cache_dir)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. "
                           f"Supported: {DatasetLoader.SUPPORTED_DATASETS}")

    @staticmethod
    def _load_sst2(max_samples: Optional[int], cache_dir: Optional[str]) -> Dict[str, Dict[str, List]]:
        """加载SST-2情感分类数据集（从本地目录）"""
        # 从项目根目录的datasets文件夹加载
        local_path = 'datasets/sst-2'

        if os.path.exists(local_path):
            # 从本地parquet文件加载
            dataset = load_dataset(local_path, trust_remote_code=False)
        else:
            raise FileNotFoundError(f"本地数据集不存在: {local_path}，请确认数据文件位置")

        train_texts = dataset['train']['sentence']
        train_labels = dataset['train']['label']
        test_texts = dataset['validation']['sentence']
        test_labels = dataset['validation']['label']

        if max_samples:
            train_texts = train_texts[:max_samples]
            train_labels = train_labels[:max_samples]
            test_texts = test_texts[:max_samples // 5]
            test_labels = test_labels[:max_samples // 5]

        return {
            'train': {'texts': train_texts, 'labels': train_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }

    @staticmethod
    def _load_ag_news(max_samples: Optional[int], cache_dir: Optional[str]) -> Dict[str, Dict[str, List]]:
        """加载AG's News新闻分类数据集"""
        try:
            dataset = load_dataset('ag_news', cache_dir=cache_dir, download_mode='reuse_cache_if_exists')
        except:
            dataset = load_dataset('C:\\Users\\ASUS\\.cache\\huggingface\\datasets\\ag_news\\default', cache_dir=cache_dir)

        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']

        if max_samples:
            train_texts = train_texts[:max_samples]
            train_labels = train_labels[:max_samples]
            test_texts = test_texts[:max_samples // 5]
            test_labels = test_labels[:max_samples // 5]

        return {
            'train': {'texts': train_texts, 'labels': train_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }

    @staticmethod
    def _load_hate_speech(max_samples: Optional[int], cache_dir: Optional[str]) -> Dict[str, Dict[str, List]]:
        """加载仇恨言论检测数据集"""
        try:
            dataset = load_dataset('hate_speech18', cache_dir=cache_dir, download_mode='reuse_cache_if_exists')

            texts = dataset['train']['text']
            labels = dataset['train']['label']

            # 二值化标签（0=无仇恨, 1=有仇恨）
            labels = [1 if l > 0 else 0 for l in labels]

            # 划分训练集和测试集
            split_idx = int(len(texts) * 0.8)
            train_texts, test_texts = texts[:split_idx], texts[split_idx:]
            train_labels, test_labels = labels[:split_idx], labels[split_idx:]

            if max_samples:
                train_texts = train_texts[:max_samples]
                train_labels = train_labels[:max_samples]
                test_texts = test_texts[:max_samples // 5]
                test_labels = test_labels[:max_samples // 5]

            return {
                'train': {'texts': train_texts, 'labels': train_labels},
                'test': {'texts': test_texts, 'labels': test_labels}
            }
        except:
            # 备用：使用简单合成数据
            print("Warning: Could not load hate_speech18, using placeholder")
            return DatasetLoader._create_placeholder_data(max_samples)

    @staticmethod
    def _load_trec(max_samples: Optional[int], cache_dir: Optional[str]) -> Dict[str, Dict[str, List]]:
        """加载TREC问题分类数据集"""
        try:
            dataset = load_dataset('trec', cache_dir=cache_dir, download_mode='reuse_cache_if_exists')
        except:
            dataset = load_dataset('C:\\Users\\ASUS\\.cache\\huggingface\\datasets\\trec\\default', cache_dir=cache_dir)

        train_texts = [f"{q['text']}" for q in dataset['train']]
        train_labels = dataset['train']['label-coarse']
        test_texts = [f"{q['text']}" for q in dataset['test']]
        test_labels = dataset['test']['label-coarse']

        if max_samples:
            train_texts = train_texts[:max_samples]
            train_labels = train_labels[:max_samples]
            test_texts = test_texts[:max_samples // 5]
            test_labels = test_labels[:max_samples // 5]

        return {
            'train': {'texts': train_texts, 'labels': train_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }

    @staticmethod
    def _load_imdb(max_samples: Optional[int], cache_dir: Optional[str]) -> Dict[str, Dict[str, List]]:
        """加载IMDB情感分类数据集"""
        try:
            dataset = load_dataset('imdb', cache_dir=cache_dir, download_mode='reuse_cache_if_exists')
        except:
            dataset = load_dataset('C:\\Users\\ASUS\\.cache\\huggingface\\datasets\\imdb\\plain_text', cache_dir=cache_dir)

        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']

        if max_samples:
            train_texts = train_texts[:max_samples]
            train_labels = train_labels[:max_samples]
            test_texts = test_texts[:max_samples // 5]
            test_labels = test_labels[:max_samples // 5]

        return {
            'train': {'texts': train_texts, 'labels': train_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }

    @staticmethod
    def _create_placeholder_data(max_samples: Optional[int]) -> Dict[str, Dict[str, List]]:
        """创建占位数据（用于测试）"""
        texts = [
            "This is a positive example.",
            "This is a negative example.",
        ] * 100

        labels = [1, 0] * 100

        if max_samples:
            texts = texts[:max_samples * 2]
            labels = labels[:max_samples * 2]

        split_idx = int(len(texts) * 0.8)

        return {
            'train': {
                'texts': texts[:split_idx],
                'labels': labels[:split_idx]
            },
            'test': {
                'texts': texts[split_idx:],
                'labels': labels[split_idx:]
            }
        }


class PoisonedDataset:
    """
    投毒数据集封装

    将干净数据集转换为投毒数据集，并支持ICL演示示例构建
    """

    def __init__(self, texts: List[str], labels: List[int],
                 attack=None, demonstrations: Optional[List[Dict]] = None):
        """
        初始化投毒数据集

        Args:
            texts: 文本列表
            labels: 标签列表
            attack: 攻击实例（可选）
            demonstrations: 预定义的演示示例
        """
        self.original_texts = texts.copy()
        self.original_labels = labels.copy()
        self.texts = texts.copy()
        self.labels = labels.copy()
        self.attack = attack
        self.demonstrations = demonstrations
        self.poison_indices = []

        if attack:
            self._apply_attack()

    def _apply_attack(self):
        """应用投毒攻击"""
        self.texts, self.labels, self.poison_indices = self.attack.poison_dataset(
            self.original_texts, self.original_labels
        )

    def create_icl_prompt(self, query_idx: int, n_shots: int = 5,
                          seed: Optional[int] = None) -> Tuple[str, List[Dict]]:
        """
        创建上下文学习(ICL)提示

        格式：
            Text: [示例1文本]
            Label: [示例1标签]

            Text: [示例2文本]
            Label: [示例2标签]
            ...
            Text: [查询文本]
            Label:

        Args:
            query_idx: 查询样本的索引
            n_shots: 演示示例数量
            seed: 随机种子

        Returns:
            prompt字符串和演示示例列表
        """
        if seed is not None:
            random.seed(seed)

        # 采样演示示例（排除查询样本）
        available_indices = [i for i in range(len(self.texts)) if i != query_idx]

        if len(available_indices) < n_shots:
            demo_indices = available_indices
        else:
            demo_indices = random.sample(available_indices, n_shots)

        # 构建演示示例
        demonstrations = []
        for idx in demo_indices:
            demonstrations.append({
                'text': self.texts[idx],
                'label': self.labels[idx],
                'is_poisoned': idx in self.poison_indices
            })

        # 构建prompt
        prompt_parts = []
        for demo in demonstrations:
            prompt_parts.append(f"Text: {demo['text']}\nLabel: {demo['label']}")

        prompt_parts.append(f"Text: {self.texts[query_idx]}\nLabel:")

        prompt = '\n\n'.join(prompt_parts)

        return prompt, demonstrations

    def get_clean_indices(self) -> List[int]:
        """获取干净样本的索引"""
        return [i for i in range(len(self.texts)) if i not in self.poison_indices]

    def get_poison_indices(self) -> List[int]:
        """获取投毒样本的索引"""
        return self.poison_indices

    def get_statistics(self) -> Dict[str, int]:
        """获取数据集统计信息"""
        return {
            'total': len(self.texts),
            'clean': len(self.texts) - len(self.poison_indices),
            'poisoned': len(self.poison_indices),
            'poison_rate': len(self.poison_indices) / len(self.texts) if self.texts else 0
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx],
            'is_poisoned': idx in self.poison_indices
        }
