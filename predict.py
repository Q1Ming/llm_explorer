# predict.py (最终版，支持分类与生成)

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification, AutoModelForCausalLM


def predict_sentiment(text, model, tokenizer):
    """处理分类任务的预测逻辑。"""
    device = model.device
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    label = model.config.id2label.get(predicted_class_id, f"UNKNOWN_ID_{predicted_class_id}")

    print(f"文本: '{text}'")
    if label == "POSITIVE":
        print(">> 预测情感: 正面 👍")
    elif label == "NEGATIVE":
        print(">> 预测情感: 负面 👎")
    else:
        print(f">> 预测结果: {label}")
    print("-" * 30)


def generate_text(prompt, model, tokenizer):
    """处理生成任务的创作逻辑。"""
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # max_length 控制生成文本的总长度
    # no_repeat_ngram_size 避免生成重复的短语
    outputs = model.generate(
        **inputs,
        max_length=60,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id  # 避免警告
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"输入: '{prompt}'")
    print(f">> 模型创作:\n{generated_text}")
    print("-" * 30)


def main():
    parser = argparse.ArgumentParser(description="使用微调好的模型进行推理。")
    parser.add_argument(
        "model_path",
        type=str,
        help="指定要加载的模型路径 (例如: outputs/sentiment_analyzer)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"错误: 模型目录 '{args.model_path}' 未找到。")
        return

    try:
        print(f"正在从 '{args.model_path}' 加载模型和分词器...")
        # 1. 先加载配置，判断模型类型
        config = AutoConfig.from_pretrained(args.model_path)

        # 2. 根据模型架构中的`architectures`字段来决定加载哪个模型类
        # 这是最可靠的方法！
        model_class = config.architectures[0]

        if "ForSequenceClassification" in model_class:
            print("检测到分类模型，进入情感分析模式...")
            model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
            task_type = "classification"
        elif "ForCausalLM" in model_class:
            print("检测到生成模型，进入文本创作模式...")
            model = AutoModelForCausalLM.from_pretrained(args.model_path)
            task_type = "generation"
        else:
            print(f"警告: 未知的模型类型 '{model_class}'。尝试使用 AutoModel 加载。")
            model = AutoModel.from_pretrained(args.model_path)
            task_type = "unknown"

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print("加载完成！")

        # 3. 将模型移动到设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"模型已移动到设备: {device.upper()}")

        # 4. 根据任务类型，启动不同的交互会话
        if task_type == "classification":
            print("\n情感分析预测已就绪。请输入一句话进行分析（输入 'exit' 或 Ctrl+C 退出）：")
            run_interactive_loop(predict_sentiment, model, tokenizer)
        elif task_type == "generation":
            print("\n文本创作已就绪。请输入开头（prompt）来让模型续写（输入 'exit' 或 Ctrl+C 退出）：")
            run_interactive_loop(generate_text, model, tokenizer)
        else:
            print("无法确定交互模式，程序退出。")

    except Exception as e:
        print(f"发生了一个意外错误: {e}")


def run_interactive_loop(handler_func, model, tokenizer):
    """一个通用的交互循环。"""
    try:
        while True:
            user_input = input("请输入: ")
            if user_input.lower() == 'exit':
                break
            if not user_input.strip():
                continue
            handler_func(user_input, model, tokenizer)
    except (KeyboardInterrupt, EOFError):
        print("\n再见！")


if __name__ == "__main__":
    main()