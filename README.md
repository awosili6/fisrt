# 基于提示擦除策略的生成式模型投毒攻击检测研究

毕业设计项目：研究基于提示擦除策略的生成式模型投毒攻击检测方法。

## 项目结构

```
project/
├── src/                          # 源代码
│   ├── attacks/                  # 攻击实现
│   │   ├── base_attack.py        # 攻击基类
│   │   ├── badnets_attack.py     # BadNets攻击
│   │   ├── insert_sent_attack.py # InsertSent攻击
│   │   └── syntactic_attack.py   # 句法触发攻击
│   ├── detectors/                # 检测策略
│   │   ├── base_detector.py      # 检测器基类
│   │   ├── prompt_eraser.py      # 提示擦除检测器
│   │   ├── attention_eraser.py   # Attention擦除检测器
│   │   └── baselines/            # 基线方法
│   │       ├── strip_detector.py # STRIP
│   │       └── onion_detector.py # ONION
│   ├── models/                   # 模型封装
│   │   └── llm_wrapper.py        # LLM统一封装
│   ├── datasets/                 # 数据处理
│   │   └── data_loader.py        # 数据集加载器
│   ├── evaluation/               # 评估模块
│   │   ├── metrics.py            # 评估指标
│   │   └── visualization.py      # 可视化
│   └── utils/                    # 工具函数
├── experiments/                  # 实验脚本
│   ├── experiment1_attack_reproduction.py      # 攻击复现
│   ├── experiment2_detection.py                # 检测对比
│   └── experiment3_sensitivity_analysis.py     # 敏感性分析
├── configs/                      # 配置文件
├── requirements.txt              # 依赖列表
└── README.md                     # 项目说明
```

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
conda create -n backdoor python=3.9
conda activate backdoor

# 安装依赖
pip install -r requirements.txt

# 安装spaCy英文模型（用于句法分析）
python -m spacy download en_core_web_sm
```

### 2. 运行实验

#### 实验1：攻击复现

```bash
# 单个实验
python experiments/experiment1_attack_reproduction.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset sst2 \
    --attack badnets \
    --poison-rate 0.1

# 参数扫描
python experiments/experiment1_attack_reproduction.py --sweep
```

#### 实验2：检测方法对比

```bash
python experiments/experiment2_detection.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset sst2 \
    --n-test 100
```

#### 实验3：参数敏感性分析

```bash
# 擦除比例敏感性
python experiments/experiment3_sensitivity_analysis.py \
    --param erase_ratio \
    --values 0.1 0.2 0.3 0.4 0.5 0.6

# 迭代次数敏感性
python experiments/experiment3_sensitivity_analysis.py \
    --param n_iterations \
    --values 1 5 10 20 30 50

# 效率-准确性权衡
python experiments/experiment3_sensitivity_analysis.py \
    --param tradeoff
```

## 核心算法说明

### 提示擦除检测策略

核心思想：通过多次随机擦除输入中的部分token，观察模型输出的变化程度。

```python
# 伪代码
function Detect(text):
    original_pred = model.predict(text)

    for i in range(n_iterations):
        erased_text = random_erase(text, erase_ratio)
        erased_pred = model.predict(erased_text)
        distance[i] = KL_Divergence(original_pred, erased_pred)

    final_score = aggregate(distance)  # mean/median/min

    return final_score > threshold
```

### 攻击方法

1. **BadNets**: 随机插入触发词（如"cf"）
2. **InsertSent**: 插入完整句子作为触发器
3. **Syntactic**: 基于句法结构的触发器

### 评估指标

- **CACC**: 干净样本准确率
- **ASR**: 攻击成功率
- **F1-Score**: 检测F1分数
- **FPR**: 误检率
- **Latency**: 检测延迟

## 实验规划

| 阶段 | 时间 | 任务 |
|------|------|------|
| 1 | 3.01-3.20 | 文献调研 |
| 2 | 3.21-4.05 | 攻击复现 |
| 3 | 4.06-4.15 | 检测框架开发 |
| 4 | 4.16-4.25 | 对比实验与优化 |
| 5 | 5.16-6.04 | 论文撰写 |

## 支持的模型

- LLaMA-2 / LLaMA-3
- ChatGLM-3
- Qwen2.5

## 支持的数据集

- SST-2 (情感分类)
- AG's News (新闻分类)
- HateSpeech (有害内容检测)
- TREC (问题分类)
- IMDB (影评分类)

## 参考资料

1. ParaFuzz: An Interpretability-driven Technique for Detecting Poisoned Samples in NLP (NeurIPS 2024)
2. BEAR: Embedding-based Adversarial Removal of Safety Backdoors (EMNLP 2024)
3. Hidden Killer: Invisible Textual Backdoor Attacks (EMNLP 2021)
4. STRIP: A Defence Against Trojan Attacks on Deep Neural Networks
5. ONION: A Simple and Effective Defense Against Textual Backdoor Attacks

## 作者信息

- 姓名：蒋杜飞
- 学号：8209220604
- 专业：软件工程2206班
- 指导教师：姚鑫
