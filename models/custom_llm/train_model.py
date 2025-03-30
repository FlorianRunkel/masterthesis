import torch
import torch.nn as nn
import torch.optim as optim
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
from models.custom_llm import SimpleGPT

VOCAB_SIZE = 5000
model = SimpleGPT(VOCAB_SIZE)

# Tokenizer laden
tokenizer = PreTrainedTokenizerFast(tokenizer_file="models/custom_llm/llm/tokenizer.json")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Beispiel-Datensätze
texts = [
    "Hallo, wie geht es dir?",
    "Künstliche Intelligenz verändert die Welt.",
    "Neuronale Netzwerke lernen aus Daten.",
]

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=50)

dataset = Dataset.from_dict({"text": texts}).map(tokenize_function, batched=True)

# 🔹 Trainings-Argumente
training_args = TrainingArguments(
    output_dir="models/custom_llm/llm",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    save_strategy="epoch",
    logging_dir="./logs",
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔹 Training
def train(model, dataset, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in dataset:
            input_ids = torch.tensor(batch["input_ids"])
            targets = input_ids.clone()
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

train(model, dataset)

torch.save(model.state_dict(), "models/custom_llm/llm/custom_llm.pth")
tokenizer.save_pretrained("models/custom_llm/llm")

print("Training abgeschlossen! Modell gespeichert.")