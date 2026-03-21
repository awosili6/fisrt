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

    def __init__(self, model, tokenizer, device: str = 'cuda', model_name: str = None):
        """
        初始化检测器

        Args:
            model: 预训练的语言模型
            tokenizer: 分词器
            device: 计算设备
            model_name: 模型名称，用于判断是否需要特殊格式化（如Qwen的ChatML）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name or ""

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

    def _format_chat_prompt(self, prompt: str) -> str:
        """
        为chat模型格式化prompt（支持Qwen的ChatML格式）
        """
        model_name_lower = self.model_name.lower()

        # Qwen2.5 ChatML格式
        if 'qwen' in model_name_lower:
            return (
                f"<|im_start|>system\n"
                f"You are a text classification assistant. "
                f"You must classify each input strictly according to the label pattern shown in the examples. "
                f"Output only the label word, nothing else.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        # Llama Chat格式
        if 'llama' in model_name_lower:
            return f"[INST] {prompt} [/INST]"

        # ChatGLM格式
        if 'chatglm' in model_name_lower:
            return f"[Round 1]\n问：{prompt}\n答："

        return prompt

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

        # 应用ChatML格式（对Qwen等chat模型）
        prompt = self._format_chat_prompt(prompt)

        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.device)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(**inputs)

            # 对于 causal LM，需要找到实际应该生成标签的位置
            # ChatML 格式下，模型在 "assistant\n" 后生成标签
            # 我们取 logits 中对应 "assistant" token 的下一个位置
            if 'qwen' in self.model_name.lower():
                # Qwen 的 ChatML 格式: ...<|im_start|>assistant
                # 需要找到 assistant token 的位置
                # 简化处理：直接生成一次，获取第一个生成 token 的分布
                # 使用 greedy 生成获取 logits
                gen_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False
                )
                # scores[0] 是第一个生成 token 的 logits
                probs = torch.softmax(gen_outputs.scores[0][0], dim=-1)
            else:
                # 获取最后一个token的logits（非ChatML模型）
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

        return probs

    def __repr__(self):
        return f"{self.__class__.__name__}(device='{self.device}')"
