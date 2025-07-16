# train_sentiment_analysis.py (用于文本生成任务)
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling


def main():
    """
    一个完整的、使用小型中文生成模型微调的流程。
    """
    # 1. 定义模型和分词器 (已更换为中文生成模型)
    model_name = "uer/gpt2-chinese-cluecorpussmall"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 关键变更：为生成任务加载模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"已加载中文生成模型: {model_name}")

    # 2. 准备数据集 (更换为中文古诗数据集)
    print("正在加载中文古诗数据集 'shibing624/chinese-poetry'...")
    dataset = load_dataset("shibing624/chinese-poetry", split="train")

    # 为了快速演示，只取一小部分数据
    small_dataset = dataset.select(range(5000))

    # 定义一个函数来对文本进行分词import torch


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
        default="uer/gpt2-chinese-cluecorpussmall",
        help="指定要加载的Hugging Face Hub上的预训练模型名称。"
    )
    args = parser.parse_args()

    try:
        print(f"正在从Hugging Face Hub加载预训练模型: {args.model_name}...")

        # 直接从Hub加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        print("加载完成！")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"模型已移动到设备: {device.upper()}")

        print("\n预训练模型已就绪。请输入开头（prompt）来让模型续写（输入 'exit' 或 Ctrl+C 退出）：")

        run_interactive_loop(model, tokenizer)

    except Exception as e:
        print(f"发生了一个意外错误: {e}")


def run_interactive_loop(model, tokenizer):
    """
    一个通用的交互循环，用于文本生成。
    """
    try:
        while True:
            prompt = input("请输入: ")
            if prompt.lower() == 'exit':
                break
            if not prompt.strip():
                continue

            device = model.device
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # 使用模型进行生成
            outputs = model.generate(
                **inputs,
                max_length=100,  # 生成更长的文本来观察效果
                num_return_sequences=1,
                no_repeat_ngram_size=2,  # 避免生成重复的词组
                pad_token_id=tokenizer.eos_token_id  # 避免警告
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"模型续写:\n{generated_text}")
            print("-" * 30)

    except (KeyboardInterrupt, EOFError):
        print("\n再见！")


if __name__ == "__main__":
    main()
    def tokenize_function(examples):
        # 古诗数据集的文本列名为 'content'
        return tokenizer(examples["content"], truncation=True, max_length=128)

    # 对整个数据集应用分词函数
    print("正在对数据集进行分词处理...")
    tokenized_datasets = small_dataset.map(tokenize_function, batched=True,
                                           remove_columns=["title", "author", "content"])

    # 3. 定义训练参数
    print("配置训练参数...")

    # 为 Tokenizer 添加 padding token，如果它没有的话
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # 使用一个特殊的数据整理器，它会为我们处理好输入和标签
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="gpt2-chinese-poetry",
        num_train_epochs=3,  # 训练 3 个 epoch
        per_device_train_batch_size=4,  # 生成任务更耗显存，减小 batch size
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
    )

    # 4. 创建并启动训练器 (Trainer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,  # 使用我们定义的数据整理器
    )

    print("开始微调中文古诗生成模型...")
    trainer.train()
    print("模型微调完成！")

    # 保存最终模型
    output_dir = "outputs/gpt2-chinese-poetry-final"
    trainer.save_model(output_dir)
    print(f"最终模型已保存至 '{output_dir}'")


if __name__ == "__main__":
    main()
