@echo off
chcp 65001 >nul
echo ========================================
echo 模型下载脚本 - 国内镜像版
echo ========================================
echo.

REM 创建模型目录
if not exist "models" mkdir models

echo [1/3] 检查 Git LFS...
git lfs version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未安装 Git LFS，请先安装: https://git-lfs.github.com/
    exit /b 1
)
echo Git LFS 已安装
echo.

echo [2/3] 设置 Hugging Face 镜像...
set HF_ENDPOINT=https://hf-mirror.com
echo 已设置 HF_ENDPOINT=%HF_ENDPOINT%
echo.

echo [3/3] 克隆 distilgpt2 模型...
if exist "models\distilgpt2" (
    echo 模型目录已存在，跳过下载
) else (
    git lfs install
    git clone https://hf-mirror.com/distilgpt2 models\distilgpt2
)
echo.

echo ========================================
echo 模型下载完成！
echo ========================================
echo.
echo 运行命令:
echo   python run.py --model ./models/distilgpt2 --quick-test --device cpu
echo.
pause
