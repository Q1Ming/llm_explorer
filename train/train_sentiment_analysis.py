# train_sentiment_analysis.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def main():
    """
    一个完整的、小型的语言模型微调流程。
    """
    # 1. 定义模型和分词器
    # 我们选用一个轻量级的预训练模型 'distilbert-base-uncased'，它又快又好用
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 加载预训练模型，并告诉它我们要做的是一个二分类任务（正面/负面）
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # --- 为模型配置标签名称 ---
    # IMDB 数据集的标签 0 代表 'negative', 1 代表 'positive'
    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    print("已为模型配置标签：0 -> NEGATIVE, 1 -> POSITIVE")

    # 2. 准备数据集
    # 加载 IMDB 电影评论数据集
    print("正在加载数据集...")
    imdb = load_dataset("imdb")

    # 定义一个函数来对文本进行分词
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # 对整个数据集应用分词函数
    print("正在对数据集进行分词处理...")
    tokenized_datasets = imdb.map(tokenize_function, batched=True)

    # 为了快速演示，我们只取一小部分数据进行训练和评估
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    # 3. 定义训练参数
    print("配置训练参数...")
    training_args = TrainingArguments(
        output_dir="test_trainer",  # 训练过程中的输出（模型检查点等）将保存在这里
        eval_strategy="steps",
        eval_steps=500,      #每 500 步评估一次
        num_train_epochs=1,  # 为了快速演示，只训练一个 epoch
        per_device_train_batch_size=8,  # 训练时的 batch size
        per_device_eval_batch_size=8,  # 评估时的 batch size

        # --- 建议的额外参数 ---
        logging_dir='./logs',  # 将日志文件保存在 logs 文件夹
        logging_steps=100,  # 每 100 步记录一次日志
        save_strategy="steps",  # 每 500 步保存一次模型检查点
        save_steps=500,
        load_best_model_at_end=True,  # 训练结束后，自动加载效果最好的模型
    )

    # 4. 创建并启动训练器 (Trainer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        tokenizer=tokenizer,  # <-- 关键补充：将分词器也交给 Trainer 管理
    )

    print("开始微调模型...")
    trainer.train()
    print("模型微调完成！")
    # 保存最终的模型和分词器
    output_dir = "outputs/sentiment_analyzer"  # 模型保存路径
    trainer.save_model(output_dir)
    print(f"最终模型已保存至 '{output_dir}'")



if __name__ == "__main__":
    main()
