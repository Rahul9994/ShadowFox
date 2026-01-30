# save as train_distilbert_imdb_gpu.py
# PREREQUISITES:
# pip install transformers datasets torch scikit-learn accelerate
# IMPORTANT: install the CUDA build of torch that matches your local CUDA (e.g. cu121):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score
import random

# -------------------------
# Configuration / hyperparams
# -------------------------
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./results"
SAVE_DIR = "./my_sentiment_model"
MAX_LENGTH = 256
TRAIN_SUBSET = 2000    # set None to use full train set
TEST_SUBSET = 500      # set None to use full test set
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
SEED = 42

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# -------------------------
# Device / GPU check
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    try:
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.current_device())
    except Exception as e:
        print(f"Error getting GPU info: {e}")
else:
    print("WARNING: CUDA is not available! Training will use CPU.")
    print("Make sure you have installed the CUDA version of PyTorch:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

# -------------------------
# Metrics
# -------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    # use weighted F1 to be robust
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1": f1}

# -------------------------
# Main
# -------------------------
def main():
    print("--- 1. Loading dataset ---")
    dataset = load_dataset("imdb")
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    # use subsets for quick runs if configured
    if TRAIN_SUBSET is not None:
        train_ds = train_ds.shuffle(seed=SEED).select(range(TRAIN_SUBSET))
    if TEST_SUBSET is not None:
        test_ds = test_ds.shuffle(seed=SEED).select(range(TEST_SUBSET))

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    print("--- 2. Tokenizer & Tokenization ---")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    # Map tokenization
    tokenized_train = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    tokenized_test = test_ds.map(tokenize_batch, batched=True, remove_columns=["text"])

    # Rename label -> labels (Trainer expects 'labels')
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    # Set torch format
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print("--- 3. Load Model ---")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Move model to GPU explicitly
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"Model moved to {device}")
        # Verify model is on GPU
        next_param = next(model.parameters())
        print(f"Model parameter device: {next_param.device}")
    else:
        print("WARNING: Model will run on CPU - CUDA not available")

    print("--- 4. TrainingArguments ---")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",   # evaluate at end of each epoch
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # enable mixed precision only if GPU is available
        dataloader_pin_memory=torch.cuda.is_available(),  # pin memory only for GPU
        report_to="none",              # disable default reporting integrations
        save_total_limit=2
    )
    
    print(f"Training will use GPU: {torch.cuda.is_available()}")
    print(f"FP16 enabled: {training_args.fp16}")

    print("--- 5. Initialize Trainer ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )
    
    # Verify trainer is using GPU
    if torch.cuda.is_available():
        # Force model to GPU one more time to be sure
        model = model.to(device)
        trainer.model = model
        # Verify model parameters are on GPU
        sample_param = next(trainer.model.parameters())
        print(f"Trainer model device: {sample_param.device}")
        print("Trainer configured to use GPU")

    print("--- 6. Start Training ---")
    trainer.train()

    print("--- 7. Final Evaluation ---")
    results = trainer.evaluate()
    print(f"Final Test Accuracy: {results.get('eval_accuracy', 0):.4f}")
    print(f"Final Test F1 Score: {results.get('eval_f1', 0):.4f}")

    print("--- 8. Save Model & Tokenizer ---")
    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)   # saves model + config
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model and tokenizer saved to {SAVE_DIR}")

    # -------------------------
    # Inference helper (GPU-safe)
    # -------------------------
    def predict_sentiment(text: str):
        # Use the trained model from trainer
        trained_model = trainer.model
        trained_model.eval()
        # Tokenize single string; return_tensors='pt'
        inputs = tokenizer(text, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = trained_model(**inputs)
            logits = outputs.logits
            predicted_id = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)[0]
            confidence = probs[predicted_id].item()
        label = "POSITIVE" if predicted_id == 1 else "NEGATIVE"
        return label, confidence

    print("\n--- 9. Test custom examples ---")
    examples = [
        "This movie was absolutely fantastic, I loved every second of it!",
        "I want my money back. The plot was boring and the acting was stiff.",
        "It was okay, not great but not terrible either."
    ]

    for text in examples:
        label, conf = predict_sentiment(text)
        print(f"Review: '{text}'\nPrediction: {label} (Confidence: {conf:.2f})\n")

if __name__ == "__main__":
    main()
