# Isaac Lab 强化学习 UI 快速训练控制台

这是一个基于 PyQt5 开发的图形化面板，旨在简化 **Isaac Lab** 强化学习任务的训练与可视化流程。
<img width="500" height="450" alt="图片" src="https://github.com/user-attachments/assets/b25be1bc-ebb1-4a17-9d92-46d8849b1818" />

## ✨ 主要功能
- **自动识别任务**：选择任务文件夹，程序自动提取 Task ID。
- **参数持久化**：训练次数 (`max_iterations`)、Conda 环境、上一次路径均可自动保存。
- **环境切换**：支持在界面上快速切换不同的 Conda 环境。
- **一键操作**：集成了开始训练、停止训练、开启 play 可视化以及自动启动 TensorBoard 的功能。
- **进程清理**：针对训练进程残留，提供一键杀掉所有 Python 进程的快捷功能。

## 🚀 使用方法

### 1. 准备工作
将 `ui_dashboard.py` 粘贴到您的 Isaac Lab 工作根目录（如 `isaaclab_files/`）下。

### 2. 安装依赖
确保您的环境中安装了 `PyQt5`：
```bash
pip install PyQt5
