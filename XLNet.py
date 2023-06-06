import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from transformers import XLNetTokenizer, XLNetForSequenceClassification


class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the XLNet model and tokenizer
model_name = 'xlnet-base-cased'
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Assuming binary classification

# Move the model to the device
model = model.to(device)

# Load the CSV file
df = pd.read_csv("C:/Users/panos/Downloads/spamdata_v2.csv")

# Split the data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.3, random_state=42,shuffle=True)

# Preprocess the data
train_texts = train_data['text'].tolist()
train_labels = train_data['label'].tolist()
test_texts = test_data['text'].tolist()
test_labels = test_data['label'].tolist()

# Set batch size and gradient accumulation steps
batch_size = 8
gradient_accumulation_steps = 4

# Create Train Dataset and DataLoader
train_dataset = MyDataset(train_texts, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create Test Dataset and DataLoader
test_dataset = MyDataset(test_texts, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Prepare optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
total_loss = 0
print("Training...")
for step, (batch_texts, batch_labels) in enumerate(train_dataloader):
    # Tokenize inputs
    encoded_inputs = tokenizer.batch_encode_plus(batch_texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)
    labels = torch.tensor(batch_labels).to(device)

    # Forward pass through the model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    total_loss += loss.item()

    # Backward pass and gradient accumulation
    loss = loss / gradient_accumulation_steps
    loss.backward()

    # Perform gradient accumulation after specified number of steps
    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    # Print progress
    if (step + 1) % 10 == 0:
        print(f"Step {step + 1}/{len(train_dataloader)}, Loss: {total_loss / (step + 1)}")

# Make predictions on the test set
# Make predictions
model.eval()
predictions = []
ground_truths = []
print("\nPredicting...")
for batch_texts, batch_labels in test_dataloader:
    # Tokenize inputs
    encoded_inputs = tokenizer.batch_encode_plus(batch_texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    batch_predictions = torch.argmax(logits, dim=1).tolist()
    predictions.extend(batch_predictions)
    ground_truths.extend(batch_labels)

# Convert predictions to labels
preds = np.array(predictions)
true_labels = np.array(ground_truths)

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, preds)
precision = precision_score(true_labels, preds, average='weighted')
recall = recall_score(true_labels, preds, average='weighted')
f1 = f1_score(true_labels, preds, average='weighted')

# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, preds))

# Compute confusion matrix
cm = confusion_matrix(true_labels, preds)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix XLNet')
plt.show()