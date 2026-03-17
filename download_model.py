#!/usr/bin/env python3
"""
模型下载脚本 - 国内镜像版
支持从多个国内源下载模型
"""

import os
import sys
import argparse
from pathlib import Path

def download_from_modelscope(model_name: str, cache_dir: str = None):
    """从 ModelScope（魔搭社区）下载模型"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("请先安装 ModelScope: pip install modelscope")
        sys.exit(1)

    print(f"从 ModelScope 下载模型: {model_name}")

    # 模型名称映射（HuggingFace -> ModelScope）
    modelscope_mapping = {
        "distilgpt2": "distilgpt2",
        "gpt2": "gpt2",
        "bert-base-uncased": "google-bert/bert-base-uncased",
        "roberta-base": "FacebookAI/roberta-base",
    }

    # 如果直接提供了 ModelScope 格式的模型名，则使用它
    if model_name.lower() in modelscope_mapping:
        model_id = modelscope_mapping[model_name.lower()]
    else:
        model_id = model_name

    print(f"ModelScope 模型ID: {model_id}")

    try:
        model_dir = snapshot_download(model_id, cache_dir=cache_dir)
        print(f"模型下载完成: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def download_from_hf_mirror(model_name: str, cache_dir: str = None):
    """从 HF-Mirror 下载模型"""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"从 HF-Mirror 下载模型: {model_name}")

    if cache_dir:
        os.environ["TRANSFORMERS_CACHE"] = cache_dir

    try:
        # 下载分词器
        print("下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 下载模型
        print("下载模型...")
        model = AutoModelForCausalLM.from_pretrained(model_name)

        print("模型下载完成!")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def download_from_wget(model_name: str, save_dir: str):
    """使用 wget 从多个源下载"""
    import urllib.request
    import json
    import ssl

    # 创建保存目录
    save_path = Path(save_dir) / model_name.replace("/", "--")
    save_path.mkdir(parents=True, exist_ok=True)

    # 需要下载的文件列表（基础文件）
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

    # 镜像源列表
    mirrors = [
        f"https://hf-mirror.com/{model_name}/resolve/main/",
        f"https://modelscope.cn/models/{model_name}/master/",
    ]

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    for file in files:
        file_path = save_path / file
        if file_path.exists():
            print(f"文件已存在，跳过: {file}")
            continue

        for mirror in mirrors:
            url = f"{mirror}{file}"
            try:
                print(f"尝试下载: {url}")
                urllib.request.urlretrieve(url, str(file_path))
                print(f"下载成功: {file}")
                break
            except Exception as e:
                print(f"  失败: {e}")
                continue
        else:
            print(f"无法下载: {file}")

    print(f"\n模型已保存到: {save_path}")
    return str(save_path)


def main():
    parser = argparse.ArgumentParser(description="下载 HuggingFace 模型")
    parser.add_argument("--model", type=str, default="distilgpt2",
                       help="模型名称，如 distilgpt2, gpt2")
    parser.add_argument("--source", type=str, default="modelscope",
                       choices=["modelscope", "hf-mirror", "wget"],
                       help="下载源")
    parser.add_argument("--cache-dir", type=str, default="./models",
                       help="模型保存目录")

    args = parser.parse_args()

    # 确保缓存目录存在
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    if args.source == "modelscope":
        model_path = download_from_modelscope(args.model, args.cache_dir)
        if model_path:
            print(f"\n使用本地模型运行命令:")
            print(f"  python run.py --model {model_path} --quick-test")

    elif args.source == "hf-mirror":
        success = download_from_hf_mirror(args.model, args.cache_dir)
        if success:
            print(f"\n模型已下载到缓存，可以直接运行:")
            print(f"  python run.py --model {args.model} --quick-test")

    elif args.source == "wget":
        model_path = download_from_wget(args.model, args.cache_dir)
        print(f"\n使用本地模型运行命令:")
        print(f"  python run.py --model {model_path} --quick-test")


if __name__ == "__main__":
    main()
