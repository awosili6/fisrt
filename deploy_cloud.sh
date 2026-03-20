#!/bin/bash
# =============================================================================
# 云端部署脚本 - 后门攻击检测项目
# 用于 AutoDL / 恒源云 / 阿里云等 GPU 云平台
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# 配置区（按需修改）
# =============================================================================
PROJECT_NAME="backdoor_detection"
PROJECT_DIR="/root/$PROJECT_NAME"
RESULTS_DIR="$PROJECT_DIR/results"
EXPERIMENTS_DIR="$PROJECT_DIR/experiments"

# 模型配置
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"  # 或 "meta-llama/Llama-2-7b-hf"
USE_8BIT=true  # 设为 true 节省显存
USE_4BIT=false

# 数据集配置
DATASETS=("sst2")  # 可添加: "ag_news" "imdb" "trec"
ATTACKS=("badnets" "insertsent" "syntactic")

# 实验配置
POISON_RATES=(0.1 0.2 0.3)
N_TEST=100  # 测试样本数

# Hugging Face Token（如需下载 gated 模型）
HF_TOKEN=""  # 如有需要，填入你的 HF token

# =============================================================================
# 函数定义
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 GPU
check_gpu() {
    log_info "检查 GPU 状态..."
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "未找到 nvidia-smi，请确认已安装 NVIDIA 驱动"
        exit 1
    fi
    
    nvidia-smi
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    log_info "检测到 GPU 显存: ${GPU_MEM}MB"
    
    if [ "$GPU_MEM" -lt 12000 ]; then
        log_warning "显存小于 12GB，建议开启 8-bit 量化"
        USE_8BIT=true
    fi
}

# 检查并设置 Hugging Face 缓存
setup_hf_cache() {
    log_info "设置 Hugging Face 缓存目录..."
    
    # AutoDL 通常有数据盘，把模型缓存放数据盘节省系统盘空间
    if [ -d "/root/autodl-tmp" ]; then
        export HF_HOME="/root/autodl-tmp/hf_cache"
        export TRANSFORMERS_CACHE="/root/autodl-tmp/hf_cache"
        log_info "使用 AutoDL 数据盘作为缓存: /root/autodl-tmp/hf_cache"
    elif [ -d "/data" ]; then
        export HF_HOME="/data/hf_cache"
        export TRANSFORMERS_CACHE="/data/hf_cache"
        log_info "使用 /data 作为缓存"
    else
        export HF_HOME="$HOME/.cache/huggingface"
        log_info "使用默认缓存目录"
    fi
    
    mkdir -p "$HF_HOME"
    log_success "缓存目录: $HF_HOME"
}

# 设置 Hugging Face Token
setup_hf_token() {
    if [ -n "$HF_TOKEN" ]; then
        log_info "配置 Hugging Face Token..."
        huggingface-cli login --token "$HF_TOKEN"
        log_success "HF Token 已配置"
    else
        log_warning "未设置 HF_TOKEN，如需要下载 gated 模型请设置"
    fi
}

# 安装系统依赖
install_system_deps() {
    log_info "安装系统依赖..."
    apt-get update -qq
    apt-get install -y -qq git-lfs vim tmux htop
    log_success "系统依赖安装完成"
}

# 安装 Python 依赖
install_python_deps() {
    log_info "安装 Python 依赖..."
    
    # 升级 pip
    pip install -q --upgrade pip
    
    # 安装 PyTorch（如未安装）
    if ! python -c "import torch" 2>/dev/null; then
        log_info "安装 PyTorch..."
        pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    
    # 安装项目依赖
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        log_info "从 requirements.txt 安装依赖..."
        pip install -q -r "$PROJECT_DIR/requirements.txt"
    else
        log_warning "未找到 requirements.txt，安装核心依赖..."
        pip install -q torch transformers datasets accelerate sentence-transformers
        pip install -q spacy nltk numpy pandas scikit-learn scipy
        pip install -q matplotlib seaborn pyyaml tqdm
    fi
    
    # 安装 spaCy 英文模型
    log_info "下载 spaCy 英文模型..."
    python -m spacy download en_core_web_sm
    
    log_success "Python 依赖安装完成"
}

# 上传/下载项目
download_project() {
    if [ -d "$PROJECT_DIR" ]; then
        log_warning "项目目录已存在: $PROJECT_DIR"
        read -p "是否重新下载? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "使用现有项目"
            return
        fi
        rm -rf "$PROJECT_DIR"
    fi
    
    log_info "下载项目..."
    
    # 方式1: 从 GitHub 克隆（推荐）
    # git clone https://github.com/yourusername/backdoor-detection.git "$PROJECT_DIR"
    
    # 方式2: 从本地上传（已手动上传）
    if [ -d "/root/project" ]; then
        mv /root/project "$PROJECT_DIR"
        log_success "项目已移动到 $PROJECT_DIR"
    elif [ -d "/root/autodl-tmp/project" ]; then
        cp -r /root/autodl-tmp/project "$PROJECT_DIR"
        log_success "项目已复制到 $PROJECT_DIR"
    else
        log_error "未找到项目文件，请手动上传项目到 /root/project 或修改脚本"
        exit 1
    fi
}

# 创建实验配置文件
create_experiment_configs() {
    log_info "创建实验配置..."
    
    mkdir -p "$PROJECT_DIR/configs/experiments"
    
    # 创建批处理实验配置
    cat > "$PROJECT_DIR/configs/experiments/batch_attack.yaml" << EOF
# 批量攻击实验配置
experiment:
  name: "batch_attack_experiments"
  seed: 42
  output_dir: "results/batch_attacks"

model:
  name: "$MODEL_NAME"
  device: "cuda"
  load_in_8bit: $USE_8BIT
  load_in_4bit: $USE_4BIT

dataset:
  name: "sst2"
  max_samples: 1000

# 批量实验参数
batch:
  attacks:
    - type: "badnets"
      trigger: "cf"
      poison_rates: [0.1, 0.2, 0.3]
    - type: "insertsent"
      trigger: "I watched this 3D movie"
      poison_rates: [0.1, 0.2, 0.3]
    - type: "syntactic"
      trigger: "S(SBAR)(,)(NP)(VP)(.)"
      poison_rates: [0.1, 0.2, 0.3]
EOF
    
    log_success "实验配置已创建"
}

# 运行单个实验
run_single_experiment() {
    local dataset=$1
    local attack=$2
    local poison_rate=$3
    
    log_info "运行实验: dataset=$dataset, attack=$attack, poison_rate=$poison_rate"
    
    cd "$PROJECT_DIR"
    
    python experiments/experiment1_attack_reproduction.py \
        --model "$MODEL_NAME" \
        --dataset "$dataset" \
        --attack "$attack" \
        --poison-rate "$poison_rate" \
        --output-dir "$RESULTS_DIR"
    
    if [ $? -eq 0 ]; then
        log_success "实验完成: ${dataset}_${attack}_${poison_rate}"
    else
        log_error "实验失败: ${dataset}_${attack}_${poison_rate}"
    fi
}

# 批量运行所有攻击实验
run_all_attack_experiments() {
    log_info "开始批量攻击实验..."
    
    for dataset in "${DATASETS[@]}"; do
        for attack in "${ATTACKS[@]}"; do
            for rate in "${POISON_RATES[@]}"; do
                run_single_experiment "$dataset" "$attack" "$rate"
            done
        done
    done
    
    log_success "所有攻击实验完成！"
}

# 运行检测实验
run_detection_experiments() {
    log_info "运行检测实验..."
    
    cd "$PROJECT_DIR"
    
    for attack in "${ATTACKS[@]}"; do
        log_info "检测实验: attack=$attack"
        
        python experiments/experiment2_detection.py \
            --model "$MODEL_NAME" \
            --dataset "${DATASETS[0]}" \
            --attack "$attack" \
            --n-test "$N_TEST" \
            --output-dir "$RESULTS_DIR"
    done
    
    log_success "检测实验完成！"
}

# 运行敏感性分析
run_sensitivity_analysis() {
    log_info "运行敏感性分析..."
    
    cd "$PROJECT_DIR"
    
    # 擦除比例敏感性
    log_info "擦除比例敏感性分析..."
    python experiments/experiment3_sensitivity_analysis.py \
        --param erase_ratio \
        --values 0.1 0.2 0.3 0.4 0.5 0.6 \
        --output-dir "$RESULTS_DIR"
    
    # 迭代次数敏感性
    log_info "迭代次数敏感性分析..."
    python experiments/experiment3_sensitivity_analysis.py \
        --param n_iterations \
        --values 1 5 10 20 30 50 \
        --output-dir "$RESULTS_DIR"
    
    log_success "敏感性分析完成！"
}

# 后台运行实验（防止 SSH 断开）
run_in_background() {
    log_info "将实验放入后台运行..."
    
    local log_file="$PROJECT_DIR/experiment_$(date +%Y%m%d_%H%M%S).log"
    
    # 创建运行脚本
    cat > "$PROJECT_DIR/run_all.sh" << 'EOF'
#!/bin/bash
cd /root/backdoor_detection
source /etc/network_turbo  # AutoDL 学术加速（如有）

# 运行所有实验
echo "=== 开始批量攻击实验 ==="
for dataset in sst2; do
    for attack in badnets insertsent syntactic; do
        for rate in 0.1 0.2 0.3; do
            echo "Running: $dataset - $attack - $rate"
            python experiments/experiment1_attack_reproduction.py \
                --model meta-llama/Llama-2-7b-chat-hf \
                --dataset "$dataset" \
                --attack "$attack" \
                --poison-rate "$rate"
        done
    done
done

echo "=== 开始检测实验 ==="
for attack in badnets insertsent syntactic; do
    python experiments/experiment2_detection.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dataset sst2 \
        --attack "$attack" \
        --n-test 100
done

echo "=== 所有实验完成 ==="
EOF
    chmod +x "$PROJECT_DIR/run_all.sh"
    
    # 使用 nohup 后台运行
    nohup "$PROJECT_DIR/run_all.sh" > "$log_file" 2>&1 &
    
    log_success "实验已在后台启动"
    log_info "日志文件: $log_file"
    log_info "查看进度: tail -f $log_file"
    log_info "查看进程: ps aux | grep experiment"
}

# 使用 tmux 运行（推荐）
run_in_tmux() {
    log_info "使用 tmux 运行实验（推荐，可随时断开重连）..."
    
    local session_name="backdoor_exp"
    
    # 检查是否已有 session
    if tmux has-session -t "$session_name" 2>/dev/null; then
        log_warning "tmux session '$session_name' 已存在"
        log_info "重新连接: tmux attach -t $session_name"
        return
    fi
    
    # 创建新的 tmux session
    tmux new-session -d -s "$session_name" -c "$PROJECT_DIR"
    
    # 在 tmux 中发送命令
    tmux send-keys -t "$session_name" "source /etc/network_turbo 2>/dev/null || true" Enter
    tmux send-keys -t "$session_name" "echo '开始批量实验...'" Enter
    tmux send-keys -t "$session_name" "bash run_all.sh" Enter
    
    log_success "tmux session 已创建: $session_name"
    log_info "连接命令: tmux attach -t $session_name"
    log_info " detach 快捷键: Ctrl+B, 然后按 D"
}

# 保存实验结果
save_results() {
    log_info "保存实验结果..."
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local result_archive="results_${timestamp}.tar.gz"
    
    cd "$PROJECT_DIR"
    
    # 打包结果
    tar -czf "$result_archive" results/ configs/ *.log 2>/dev/null || true
    
    log_success "结果已打包: $PROJECT_DIR/$result_archive"
    
    # 显示结果统计
    log_info "实验结果统计:"
    find "$RESULTS_DIR" -name "*.json" -o -name "*.csv" -o -name "*.png" 2>/dev/null | head -20
}

# 监控实验状态
monitor_status() {
    log_info "当前实验状态:"
    echo "================================"
    echo "GPU 状态:"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=table
    echo ""
    echo "运行中的 Python 进程:"
    ps aux | grep python | grep -v grep || echo "无 Python 进程运行"
    echo ""
    echo "磁盘使用:"
    df -h | grep -E "(Filesystem|/root|/data)"
    echo ""
    echo "实验结果:"
    ls -lh "$RESULTS_DIR" 2>/dev/null | tail -10 || echo "暂无结果"
    echo "================================"
}

# 清理环境
cleanup() {
    log_warning "清理环境..."
    
    read -p "确定要清理所有实验结果吗? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$RESULTS_DIR"/*
        rm -f "$PROJECT_DIR"/*.log
        rm -f "$PROJECT_DIR"/*.tar.gz
        log_success "清理完成"
    else
        log_info "取消清理"
    fi
}

# =============================================================================
# 主菜单
# =============================================================================

show_menu() {
    echo ""
    echo "=================================="
    echo "  后门攻击检测 - 云端部署脚本"
    echo "=================================="
    echo "1. 完整部署（环境+依赖+运行）"
    echo "2. 仅安装环境依赖"
    echo "3. 下载/更新项目"
    echo "4. 运行批量攻击实验"
    echo "5. 运行检测实验"
    echo "6. 运行敏感性分析"
    echo "7. 后台运行所有实验 (tmux)"
    echo "8. 查看实验状态"
    echo "9. 打包下载结果"
    echo "10. 清理环境"
    echo "0. 退出"
    echo "=================================="
}

# 完整部署
full_deploy() {
    log_info "开始完整部署..."
    
    check_gpu
    setup_hf_cache
    install_system_deps
    download_project
    install_python_deps
    create_experiment_configs
    
    log_success "部署完成！项目路径: $PROJECT_DIR"
    log_info "建议先测试: python $PROJECT_DIR/run_mock.py"
    log_info "然后运行: cd $PROJECT_DIR && python experiments/experiment1_attack_reproduction.py --help"
}

# 主函数
main() {
    # 如果是直接运行（带参数），执行对应功能
    case "${1:-}" in
        deploy)
            full_deploy
            ;;
        install)
            check_gpu
            setup_hf_cache
            install_python_deps
            ;;
        run)
            run_all_attack_experiments
            ;;
        detect)
            run_detection_experiments
            ;;
        sensitivity)
            run_sensitivity_analysis
            ;;
        bg|background)
            run_in_tmux
            ;;
        status)
            monitor_status
            ;;
        save)
            save_results
            ;;
        clean)
            cleanup
            ;;
        *)
            # 交互式菜单
            while true; do
                show_menu
                read -p "请选择操作 [0-10]: " choice
                
                case $choice in
                    1) full_deploy ;;
                    2) 
                        check_gpu
                        setup_hf_cache
                        install_python_deps
                        ;;
                    3) download_project ;;
                    4) run_all_attack_experiments ;;
                    5) run_detection_experiments ;;
                    6) run_sensitivity_analysis ;;
                    7) run_in_tmux ;;
                    8) monitor_status ;;
                    9) save_results ;;
                    10) cleanup ;;
                    0) 
                        log_info "退出脚本"
                        exit 0
                        ;;
                    *)
                        log_error "无效选项"
                        ;;
                esac
                
                echo ""
                read -p "按回车继续..."
            done
            ;;
    esac
}

# 运行主函数
main "$@"
