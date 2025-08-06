import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict


def main():
    """
    主函数，负责解析参数、加载模型和启动交互循环。
    """
    parser = argparse.ArgumentParser(
        description="直接加载和运行一个大语言模型进行交互式对话。",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息中的换行
    )

    parser.add_argument(
        "--model_name",
        type=str,
        # 推荐使用一个响应速度快的聊天模型作为默认值
        default="Qwen/Qwen1.5-0.5B-Chat",
        help="指定要加载的模型标识符或本地路径。\n"
             "示例:\n"
             "  - 从Hugging Face Hub加载: 'Qwen/Qwen1.5-0.5B-Chat'\n"
             "  - 从本地加载: 'C:/My/File/Models/Qwen3-0.6B'"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,  # 默认值设为256，在CPU上性能较好
        help="控制模型生成回复的最大长度(token数)。\n"
             "在CPU上，较小的值(如256)响应更快，较大的值(如1024)会很慢。"
    )

    parser.add_argument(
        "--no_sample",
        action="store_true",  # 作为一个开关使用，出现即为True
        help="禁用采样，使用贪心搜索(greedy search)进行解码。\n"
             "这会让模型的回答更确定，同时在CPU上可以显著提速。"
    )

    # 添加采样相关参数
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度。值越高，输出越随机。仅在启用采样时有效。"
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="核采样参数。值越低，输出越确定。仅在启用采样时有效。"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-K采样参数。值越低，输出越确定。仅在启用采样时有效。"
    )

    args = parser.parse_args()

    try:
        print(f"准备从 '{args.model_name}' 加载模型和分词器...")

        # 加载分词器，trust_remote_code=True对于新模型是必需的
        # use_fast=True 会尽可能使用Rust实现的高速分词器，以提升性能
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            use_fast=True
        )

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype="auto",  # 自动选择最佳的数据类型 (如bfloat16)
            device_map="auto",  # 自动将模型分配到最合适的设备上 (CPU, GPU, MPS)
            trust_remote_code=True  # 关键参数：允许执行模型仓库中的自定义代码，以加载非官方库内置的模型架构
        )
        print("模型加载完成！")

        # 明确告知用户模型运行在哪个设备上
        print(f"模型已成功加载到设备: {model.device.type.upper()}")

        # 如果在CPU上运行，且用户没有指定--no_sample，给一个性能提示
        if model.device.type == 'cpu' and not args.no_sample:
            print("\n性能提示: 当前在CPU上运行。为了获得更快的响应速度，")
            print("          可以尝试使用 '--no_sample' 参数重新启动程序。")

        print("\n" + "=" * 50)
        print(" LLM Explorer 已就绪")
        print("=" * 50)
        print("与模型开始对话吧！(输入 'exit' 或按 Ctrl+C 退出)")

        # 启动交互循环，并传入解析后的参数
        run_interactive_loop(model, tokenizer, args.max_new_tokens, args.no_sample,
                             args.temperature, args.top_p, args.top_k)

    except Exception as e:
        print(f"\n[错误] 发生了一个意外错误: {e}")
        # 在网络错误时提供更明确的建议
        if "Connection" in str(e) or "TimedOut" in str(e):
            print("错误提示: 这是一个网络连接错误。")
            print("请检查您的网络连接、代理或防火墙设置。")
            print(f"如果您想从本地加载，请确保路径 '{args.model_name}' 是正确的。")
        # 在模型文件不完整时提供建议
        elif "trust_remote_code" in str(e):
            print("错误提示: 无法加载模型架构。")
            print("1. 请确保您的模型文件是完整的，特别是对于需要自定义代码的模型，需包含相关的.py文件。")
            print("2. 尝试清理Hugging Face缓存，然后让程序自动重新下载。")
            print(r"   Windows缓存路径通常在: C:\Users\<你的用户名>\.cache\huggingface")


def run_interactive_loop(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                         max_new_tokens: int, no_sample: bool,
                         temperature: float, top_p: float, top_k: int):
    """
    一个通用的交互循环，使用聊天模板与模型对话，支持多轮历史记录。

    Args:
        model: 已加载的transformers模型。
        tokenizer: 已加载的transformers分词器。
        max_new_tokens: 控制生成长度的参数。
        no_sample: 是否禁用采样（使用贪心搜索）。
        temperature: 采样温度。
        top_p: 核采样参数。
        top_k: Top-K采样参数。
    """
    # 维护一个符合聊天模板格式的对话历史
    # 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    chat_history: List[Dict[str, str]] = []

    try:
        while True:
            user_input = input("你: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():  # 忽略空输入
                continue

            # 将用户的最新输入添加到对话历史
            chat_history.append({"role": "user", "content": user_input})

            # 关键步骤：使用聊天模板 (Chat Template)
            # 这是与现代聊天模型正确交互的核心。它会将整个对话历史（包括用户和助手的回合）
            # 格式化成模型在训练时所见的、包含特殊token（如<|im_start|>、<|im_end|>）的字符串。
            # 直接拼接字符串是错误的做法。
            prompt = tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True  # 为助手角色添加起始提示，引导模型开始生成
            )

            # 将格式化后的prompt编码成输入张量，并移动到模型所在的设备
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # 准备生成参数 - 只添加需要的参数以避免警告
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": not no_sample,  # 如果no_sample为True，则do_sample为False，即贪心搜索
                "pad_token_id": tokenizer.eos_token_id  # 避免在批处理生成时出现pad_token_id未设置的警告
            }

            # 只在启用采样时添加采样参数
            if not no_sample:
                generation_kwargs.update({
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                })

            # 生成回复
            outputs = model.generate(**inputs, **generation_kwargs)

            # 解码时，需要从输出张量中排除掉我们输入的prompt部分。
            # outputs[0] 是完整的生成序列，包括输入。
            # inputs['input_ids'].shape[-1] 是输入序列的长度。
            # 通过切片操作，我们只解码新生成的部分。
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

            # 过滤掉模型的“思考”过程
            # 使用正则表达式移除 <think>...</think> 标签及其内容
            # cleaned_response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL).strip()

            # 为了对话历史的一致性，我们将清理后的版本赋给response
            # response = cleaned_response

            print(f"模型: {response}")
            print("-" * 30)

            # 将模型的回复也添加到对话历史中，以便进行下一轮有上下文的对话
            chat_history.append({"role": "assistant", "content": response})

    except (KeyboardInterrupt, EOFError):
        # 优雅地处理Ctrl+C或Ctrl+D退出
        print("\n\n对话结束，再见！")


if __name__ == "__main__":
    main()