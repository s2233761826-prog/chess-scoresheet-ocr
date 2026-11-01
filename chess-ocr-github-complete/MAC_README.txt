Mac 启动说明

本包包含用户提供的识别模型、检测模型及原样保留的字符字典。
两套模型已分别放入 models/finetuned/rec_chess_infer 和 det_chess_infer。

1. 安装 Python 3.11（不必卸载现有 Python 3.13）。
2. 在终端输入 bash 后加空格，将 start_mac.command 拖入终端，再按回车。
3. 首次运行会创建独立环境并安装依赖，需联网。以后使用同一脚本启动。
4. 打开终端显示的 Local URL，通常是 http://localhost:8501。
5. 保持终端开启；按 Ctrl+C 停止服务。

此次修改：新增 Mac 启动脚本；明确禁用 GPU 与 MKLDNN；加入两套模型及字典。
已验证文件复制完整性、Python 语法及启动脚本语法。
尚未在 Apple 芯片 Mac 上执行依赖安装或模型推理，不能据此保证识别效果。
原 README 中“不包含模型”指此前的纯源码版本，本地运行包已附模型。
