"""
投毒检测器基类定义
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseDetector(ABC):
    """
    投毒检测器基类

    所有检测器必须继承此类并实现detect方法
    """

    def __init__(self, model, tokenizer, device: str = 'cuda'):
        """
        初始化检测器

        Args:
            model: 预训练的语言模型
            tokenizer: 分词器
            device: 计算设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @abstractmethod
    def detect(self, text: str, demonstrations: List[Dict] = None) -> Dict[str, Any]:
        """
        检测单个样本是否为投毒样本

        Args:
            text: 待检测的文本
            demonstrations: ICL演示示例列表，每项包含'text', 'label', 'is_poisoned'等字段

        Returns:
            检测结果字典，包含：
            - is_poisoned: bool，是否为投毒样本
            - score: float，异常分数（越高越可能是投毒样本）
            - confidence: float，置信度
        """
        pass

    def batch_detect(self, texts: List[str], demonstrations: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        批量检测

        Args:
            texts: 待检测的文本列表
            demonstrations: ICL演示示例列表

        Returns:
            检测结果列表
        """
        return [self.detect(text, demonstrations) for text in texts]

    def compute_prediction(self, text: str, demonstrations: List[Dict] = None):
        """
        辅助方法：获取模型的预测分布

        Args:
            text: 输入文本
            demonstrations: ICL演示示例

        Returns:
            预测概率分布 (torch.Tensor)
        """
        import torch

        # 构建完整的ICL prompt
        if demonstrations:
            prompt_parts = []
            for demo in demonstrations:
                prompt_parts.append(f"Text: {demo['text']}\nLabel: {demo['label']}")
            prompt_parts.append(f"Text: {text}\nLabel:")
            prompt = '\n\n'.join(prompt_parts)
        else:
            prompt = text

        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.device)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取最后一个token的logits
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        return probs

    def __repr__(self):
        return f"{self.__class__.__name__}(device='{self.device}')"
