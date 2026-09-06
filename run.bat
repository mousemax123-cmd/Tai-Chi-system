@echo off
echo 正在启动太极拳动作识别与评分系统...
echo.

:: 检查Python是否已安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Python安装。
    echo 请先安装Python 3.8或更高版本，然后再运行此程序。
    echo 您可以从 https://www.python.org/downloads/ 下载Python。
    pause
    exit /b 1
)

:: 检查必要的库是否已安装
echo 检查必要的库...
pip show mediapipe >nul 2>&1
if %errorlevel% neq 0 (
    echo 安装必要的库...
    pip install mediapipe opencv-python numpy matplotlib fastdtw scipy streamlit
    if %errorlevel% neq 0 (
        echo 错误: 安装库失败。
        pause
        exit /b 1
    )
)

:: 创建必要的文件夹
if not exist data\standard mkdir data\standard
if not exist data\student mkdir data\student
if not exist data\processed mkdir data\processed
if not exist results mkdir results

echo 所有准备工作已完成！
echo 启动应用程序...
echo.
echo 提示: 应用启动后，请在浏览器中访问 http://localhost:8501
echo.

:: 启动应用
streamlit run app.py

pause