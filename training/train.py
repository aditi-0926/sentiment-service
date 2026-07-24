from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np
import json
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/final"

def main():
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    # Subset for a fast first dry run — remove these two lines later for full training
    dataset["train"] = dataset["train"].select(range(2000))
    dataset["validation"] = dataset["validation"].select(range(500))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir="models/checkpoint",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate(dataset["test"])

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete. Metrics:", metrics)

if __name__ == "__main__":
    main()