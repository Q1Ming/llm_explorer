import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    """
    直接加载一个预训练的生成模型并进行交互，不经过任何微调。
    """
    parser = argparse.ArgumentParser(description="直接使用预训练的生成模型进行推理。")

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen1.5-0.5B-Chat",
        help="指定要加载的Hugging Face Hub上的预训练模型名称。"
    )
    args = parser.parse_args()

    try:
        print(f"正在从Hugging Face Hub加载预训练模型: {args.model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            # 对于小显存设备，可以考虑使用低精度加载
            torch_dtype="auto"
        )
        print("加载完成！")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"模型已移动到设备: {device.upper()}")

        print("\n预训练模型已就绪。请输入内容与模型对话（输入 'exit' 或 Ctrl+C 退出）：")

        run_interactive_loop(model, tokenizer)

    except Exception as e:
        print(f"发生了一个意外错误: {e}")
        # 在网络错误时提供更明确的建议
        if "Connection" in str(e) or "TimedOut" in str(e):
            print("提示: 这是一个网络连接错误。请检查您的网络连接、代理或防火墙设置。")


def run_interactive_loop(model, tokenizer):
    """
    一个通用的交互循环，使用聊天模板与模型对话。
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

            # 1. 将用户输入添加到对话历史
            chat_history.append({"role": "user", "content": user_input})

            device = model.device

            # 2. ★ 关键优化：使用聊天模板 ★
            # 将整个对话历史格式化成模型能理解的字符串
            prompt = tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            tokens = 1024
            # 3. 使用模型进行生成
            outputs = model.generate(
                **inputs, # 这样会将 input_ids 和 attention_mask 都传入
                max_new_tokens=tokens,  # 限制新生成token的数量
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            # 4. 解码生成的回复
            # 只解码新生成的部分，排除掉输入的prompt
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

            print(f"模型: {response}")
            print("-" * 30)

            # 5. 将模型的回复也添加到对话历史中，以便进行多轮对话
            chat_history.append({"role": "assistant", "content": response})

    except (KeyboardInterrupt, EOFError):
        print("\n再见！")


if __name__ == "__main__":
    main()