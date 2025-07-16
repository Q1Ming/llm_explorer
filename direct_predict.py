import torch
import argparse
import re  # 导入正则表达式库，用于文字处理
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    """
    直接加载一个预训练的生成模型并进行交互，不经过任何微调。
    """
    parser = argparse.ArgumentParser(description="直接使用预训练的生成模型进行推理。")

    parser.add_argument(
        "--model_name",
        type=str,
        # 将你常用的本地模型路径设为默认值
        default="Qwen/Qwen1.5-0.5B-Chat",
        help="指定要加载的模型路径。\n示例:\n"
             "  - 从Hub加载: Qwen/Qwen1.5-0.5B-Chat\n"
             "  - 从本地加载: C:/My/File/Models/Qwen3-0.6B"
    )

    # 将 max_new_tokens 做成可配置参数
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="控制模型生成回复的最大长度(token数)。\n"
             "在CPU上，较小的值(如128)响应更快，较大的值(如1024)会很慢。"
    )
    args = parser.parse_args()

    try:
        # 智能判断是加载本地模型还是Hub模型
        # (这个逻辑可以简化，因为`from_pretrained`会自动处理)
        print(f"准备从 '{args.model_name}' 加载模型...")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            # 对CPU同样有效，确保以合适的格式加载
            torch_dtype="auto",
            # 自动将模型放在最合适的设备上（CPU或GPU）
            device_map="auto"
        )
        print("加载完成！")

        # device_map="auto" 已经完成了设备放置
        print(f"模型已加载到设备: {model.device.type.upper()}")

        print("\n预训练模型已就绪。请输入内容与模型对话（输入 'exit' 或 Ctrl+C 退出）：")

        run_interactive_loop(model, tokenizer, args.max_new_tokens)

    except Exception as e:
        print(f"发生了一个意外错误: {e}")
        # 在网络错误时提供更明确的建议
        if "Connection" in str(e) or "TimedOut" in str(e):
            print("提示: 这是一个网络连接错误。请检查您的网络连接、代理或防火墙设置。")


def run_interactive_loop(model, tokenizer, max_new_tokens):
    """
    一个通用的交互循环，使用聊天模板与模型对话，并进行智能后处理。
    """
    # 维护一个对话历史
    chat_history = []

    try:
        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                break
            if not user_input.strip():
                continue

            #  将用户输入添加到对话历史
            chat_history.append({"role": "user", "content": user_input})

            device = model.device

            # 关键优化：使用聊天模板
            # 将整个对话历史格式化成模型能理解的字符串
            prompt = tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # 关键参数调优与修正
            outputs = model.generate(
                **inputs,  # 正确传递 input_ids 和 attention_mask
                # 使用命令行传入的参数
                max_new_tokens=max_new_tokens,
                # 对于CPU，关闭采样使用贪心搜索，会显著提速 即：do_sample=False
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # 只解码新生成的部分，排除掉输入的prompt，得到原始回复
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

            # 过滤掉模型的“思考”过程
            # 使用正则表达式移除 <think>...</think> 标签及其内容
            # cleaned_response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL).strip()

            # 为了对话历史的一致性，我们将清理后的版本赋给response
            # response = cleaned_response

            print(f"模型: {response}")
            print("-" * 30)

            # 5. 将模型的回复也添加到对话历史中，以便进行多轮对话
            chat_history.append({"role": "assistant", "content": response})

    except (KeyboardInterrupt, EOFError):
        print("\n再见！")


if __name__ == "__main__":
    main()
