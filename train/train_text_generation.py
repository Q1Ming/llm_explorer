import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedModel
)


def main() -> None:
    """
    一个完整的、使用Hugging Face Transformers库微调中文文本生成模型的流程。
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

        # 1. 定义模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        print("加载完成！")

        # 2. 准备数据集 (更换为中文古诗数据集)
        print("正在加载中文古诗数据集 'shibing624/chinese-poetry'...")
        dataset = load_dataset("shibing624/chinese-poetry", split="train")

        # 为了快速演示，只取一小部分数据
        small_dataset = dataset.select(range(5000))

        def tokenize_function(examples: dict) -> dict:
            # 古诗数据集的文本列名为 'content'
            return tokenizer(examples["content"], truncation=True, max_length=128, padding="max_length")

        # 对整个数据集应用分词函数
        print("正在对数据集进行分词处理...")
        tokenized_datasets = small_dataset.map(tokenize_function, batched=True,
                                               remove_columns=["title", "author", "content"])

        # 3. 定义训练参数
        print("配置训练参数...")

        # 为 Tokenizer 添加 padding token，如果它没有的话
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
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
            fp16=torch.cuda.is_available(),  # 如果有GPU，开启半精度训练加速
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
        tokenizer.save_pretrained(output_dir)  # 同样保存tokenizer
        print(f"最终模型和分词器已保存至 '{output_dir}'")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")


if __name__ == "__main__":
    main()
