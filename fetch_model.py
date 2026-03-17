#!/usr/bin/env python3
"""
手动下载 HuggingFace 模型文件（绕过 LFS）
"""

import os
import ssl
import urllib.request
from pathlib import Path

# 禁用 SSL 验证（如果必要）
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url: str, save_path: Path):
    """下载单个文件"""
    if save_path.exists():
        print(f"  已存在: {save_path.name}")
        return True

    try:
        print(f"  下载: {save_path.name} ...", end=" ")
        urllib.request.urlretrieve(url, str(save_path))
        print(f"完成 ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"失败: {e}")
        if save_path.exists():
            save_path.unlink()
        return False


def download_model(model_name: str = "distilgpt2", save_dir: str = "./models"):
    """下载完整模型"""

    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"下载模型: {model_name}")
    print(f"保存到: {save_path}")
    print()

    # 镜像源
    base_url = f"https://hf-mirror.com/{model_name}/resolve/main/"

    # 需要下载的文件
    files = [
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]

    success_count = 0
    for file in files:
        url = base_url + file
        if download_file(url, save_path / file):
            success_count += 1

    print(f"\n下载完成: {success_count}/{len(files)} 个文件")
    print(f"\n使用命令:")
    print(f"  python run.py --mode attack --model {save_path} --quick-test --device cpu")

    return str(save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--save-dir", default="./models")
    args = parser.parse_args()

    download_model(args.model, args.save_dir)
