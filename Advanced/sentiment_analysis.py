# ==========================================
# PREREQUISITES
# Run this command in your terminal before running the script:
# pip install transformers datasets torch scikit-learn accelerate
# ==========================================

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score
import os

# Limit CPU threads to reduce CPU usage (set to 2-4 threads instead of all cores)
# Adjust this number based on your CPU cores (e.g., 2 for 4-core CPU, 4 for 8-core CPU)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)  # Limit PyTorch threads

def compute_metrics(eval_pred):
    """
    Callback function to calculate accuracy and F1 score 
    during evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

def main():
    print("--- 1. Loading and Preprocessing Data ---")
    
    # Load IMDb dataset
    # Reduced dataset size to lower CPU usage
    # We are using a smaller subset (1000 train, 250 test) to reduce CPU load
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))  # Reduced from 2000
    test_dataset = dataset["test"].shuffle(seed=42).select(range(250))     # Reduced from 500

    # Initialize Tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Tokenization function - optimized for CPU
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True,
            max_length=128  # Reduced from default 512 to speed up processing
        )

    # Apply tokenization - optimized for CPU
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=100,  # Process in smaller batches
        num_proc=1       # Use single process to reduce CPU load
    )
    tokenized_test = test_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=100,
        num_proc=1
    )

    print("--- 2. Initializing Model ---")
    # Load DistilBERT with a classification head on top
    # num_labels=2 corresponds to Binary Classification (Positive/Negative)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("--- 3. Setting up Training ---")
    # Define training arguments - optimized for CPU to reduce load
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",      # Check accuracy at end of every epoch
        save_strategy="epoch",      # Save model at end of every epoch
        learning_rate=2e-5,
        per_device_train_batch_size=4,   # Reduced from 16 to 4 for CPU
        per_device_eval_batch_size=4,   # Reduced from 16 to 4 for CPU
        num_train_epochs=2,         # Reduced from 3 to 2 epochs
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
        dataloader_num_workers=0,   # Disable multiprocessing to reduce CPU load
        dataloader_pin_memory=False, # Disable pin memory for CPU
        dataloader_drop_last=True,  # Drop incomplete batches
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    print("--- 4. Starting Training (Fine-Tuning) ---")
    trainer.train()

    print("--- 5. Final Evaluation ---")
    results = trainer.evaluate()
    print(f"Final Test Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Final Test F1 Score: {results['eval_f1']:.4f}")

    # Save the final model locally
    model.save_pretrained("./my_sentiment_model")
    tokenizer.save_pretrained("./my_sentiment_model")
    print("Model saved to ./my_sentiment_model")

    # ==========================================
    # CUSTOM INFERENCE
    # ==========================================
    print("\n--- 6. Testing Custom Examples ---")
    
    def predict_sentiment(text):
        # Prepare inputs
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        # Move inputs to same device as model (GPU/CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits
        
        predicted_class_id = logits.argmax().item()
        
        # Map ID to Label (0 = Negative, 1 = Positive for IMDb)
        label = "POSITIVE" if predicted_class_id == 1 else "NEGATIVE"
        confidence = torch.softmax(logits, dim=1)[0][predicted_class_id].item()
        
        return label, confidence

    # Let's try some examples
    examples = [
        "This movie was absolutely fantastic, I loved every second of it!",
        "I want my money back. The plot was boring and the acting was stiff.",
        "It was okay, not great but not terrible either."
    ]

    for text in examples:
        label, score = predict_sentiment(text)
        print(f"Review: '{text}'\nPrediction: {label} (Confidence: {score:.2f})\n")

if __name__ == "__main__":
    main()