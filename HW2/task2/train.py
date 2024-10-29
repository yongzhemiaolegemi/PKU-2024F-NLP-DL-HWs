import logging
import wandb
from transformers import (
    HfArgumentParser, 
    set_seed, 
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from dataclasses import dataclass, field
from typing import Optional
from dataHelper import get_dataset  # 假设已经定义的 get_dataset 函数
from evaluate import load
import os
os.environ["HF_HUB_BASE_URL"] = "https://hf-mirror.com"
# 定义自定义参数类来包含 model_name_or_path 参数
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="roberta-base", metadata={"help": "The model checkpoint for weights initialization."}
    )
    dataset: Optional[str] = field(
        default="restaurant_sup", metadata={"help": "The dataset to use."}
    )

# 使用 HfArgumentParser 解析参数
parser = HfArgumentParser((ModelArguments, TrainingArguments))
model_args, training_args = parser.parse_args_into_dataclasses()
training_args.report_to = ["wandb"]
# 初始化 WandB
wandb.init(project="my-huggingface-project", name="experiment_name"+model_args.model_name_or_path+"_"+model_args.dataset)

# 设置静态参数，避免在命令行传入
output_dir = "./results"
batch_size = 16
num_epochs = 3
import random
seed = random.randint(0, 1000)  


# 定义 TrainingArguments
training_args.output_dir = output_dir
training_args.per_device_train_batch_size = batch_size
training_args.per_device_eval_batch_size = batch_size
training_args.num_train_epochs = num_epochs
training_args.seed = seed
training_args.report_to = ["wandb"]

# 初始化日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("Initializing training script")

# 设置随机种子以确保可复现性
set_seed(training_args.seed)

# 加载数据集
logger.info("Loading dataset...")
dataset = get_dataset(model_args.dataset, sep_token="[SEP]")
unique_labels = set(dataset["train"]["label"])
label_count = len(unique_labels)
train_dataset, eval_dataset = dataset["train"], dataset["test"]

# 加载模型配置、tokenizer 和模型
logger.info("Loading model and tokenizer...")
config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=label_count)  # 根据数据集调整 num_labels
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)

# 预处理数据集：对文本进行分词
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# 加载评估指标
accuracy = load("accuracy")
f1_metric = load("f1")

# 定义计算指标的函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_micro = f1_metric.compute(predictions=preds, references=labels, average="micro")
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "micro_f1": f1_micro["f1"], "macro_f1": f1_macro["f1"]}

# 初始化 DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 使用 Trainer 进行训练和评估
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)

# 开始训练和评估
if training_args.do_train:
    logger.info("Starting training...")
    trainer.train()

if training_args.do_eval:
    logger.info("Starting evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

logger.info("Training complete.")

# 结束 WandB 会话
wandb.finish()
