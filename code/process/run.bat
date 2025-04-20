@echo off

:: 设置虚拟环境路径和脚本路径
set VENV_PATH=E:\uni\code\Python\CoT-RL\.venv
set SCRIPT_PATH=E:\uni\code\Python\CoT-RL\code\process\process.py

:: 检查虚拟环境是否存在
if not exist "%VENV_PATH%\Scripts\python.exe" (
    echo Error: Virtual environment not found at: %VENV_PATH%
    pause
    exit /b 1
)

:loop
:: 启动 Python 脚本
echo Starting the Python Script...
"%VENV_PATH%\Scripts\python.exe" "%SCRIPT_PATH%"

:: 检查退出状态码
if %errorlevel% equ 0 (
    echo Script exited successfully. Stopping the loop.
    pause
    exit /b 0
) else (
    echo Script crashed (Exit Code: %errorlevel%). Restarting in 5 seconds...
    timeout /t 5 >nul
    goto loop
)
