# LLM Explorer

一个用于加载和交互式运行各种大语言模型的简单项目。

## ✨ 特性

- 💻 **轻松交互**: 提供一个简单的命令行界面，用于与聊天模型进行多轮对话。
- 🌐 **模型来源灵活**: 支持从 Hugging Face Hub 在线加载模型，或从本地磁盘加载已下载的模型。
- 🚀 **CPU 优化**: 专为在 CPU 上运行而优化，并提供性能提示以加快响应速度。
- 📦 **环境纯净**: 使用 Poetry 进行依赖管理，确保在任何机器上都能一键构建可复现的运行环境。
- 🛠️ **高度可配置**: 可通过命令行参数轻松切换模型、调整生成长度和解码策略。

## 🔧 环境准备

在开始之前，请确保你的系统已安装以下软件：

1.  **Python 3.12 或更高版本**
2.  **Poetry**: Python 依赖管理工具。
3.  **Git**: 用于克隆本项目。

### ❗ Windows 系统特别说明

在 Windows 上运行本项目前，你需要安装 **Microsoft Visual C++ Redistributable**。否则，在导入 `torch` 时可能会遇到 `WinError 126` 错误。

- 点击此处从微软官方网站下载 (选择 X64 版本)
- **重要提示**: 安装后请务必**重启电脑**，以确保系统路径生效。

## 🚀 快速开始

1.  **克隆项目**
    ```sh
    git clone https://github.com/Q1Ming/llm_explorer.git
    cd llm_explorer
    ```

2.  **安装依赖**
    运行以下命令。Poetry 会自动读取 `pyproject.toml` 文件，在项目根目录下创建一个 `.venv` 虚拟环境，并安装所有必需的依赖包。
    ```sh
    poetry install
    ```

## 💬 如何运行

所有命令都应在项目根目录下执行。

### 示例 1: 加载默认模型 (从 Hugging Face Hub)

这是最简单的启动方式。程序将自动从网络下载并加载 `Qwen/Qwen1.5-0.5B-Chat` 模型。
```sh
poetry run python direct_predict.py
```
> 第一次运行时会下载模型，可能需要一些时间。后续运行会直接从本地缓存加载。

### 示例 2: 加载本地模型

如果你已经将模型下载到本地，可以使用 `--model_name` 参数指定其路径。
```sh
# 路径请使用引号包裹，并推荐使用正斜杠 "/"
poetry run python direct_predict.py --model_name "C:/My/File/Models/Qwen3-0.6B"
```

### 示例 3: 提升 CPU 响应速度

在 CPU 上运行时，使用 `--no_sample` 参数可以采用贪心搜索策略，这会使模型响应速度显著加快。
```sh
poetry run python direct_predict.py --no_sample
```

### 示例 4: 调整生成长度

使用 `--max_new_tokens` 控制模型单次回复的最大长度。
```sh
poetry run python direct_predict.py --max_new_tokens 512
```

## 🙏 致谢

本项目作为一个入门级实践，旨在探索如何在本地调用大语言模型。其开发过程得到了 **Gemini Code Assist** 与 **通义灵码 (Qwen3-Coder)** 的辅助。
```