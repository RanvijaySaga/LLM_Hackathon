import os
import torch
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

from com.abnamro.llm.constants import Constants


# Custom Dataset Class
class ExpenseDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_len):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, index):
        description = str(self.descriptions[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Training Loop
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_preds = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_preds += torch.sum(preds == labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return correct_preds.double() / len(data_loader.dataset), total_loss / len(data_loader)


# Validation Loop
def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_preds = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_preds += torch.sum(preds == labels)
            total_loss += loss.item()

    return correct_preds.double() / len(data_loader.dataset), total_loss / len(data_loader)


# Function to Train the Model
def train_model(csv_file):
    df = pd.read_csv(csv_file)

    # Encode ExpenseType using LabelEncoder
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['ExpenseType'])
    print(df)

    # Split the data into training and validation sets
    # train_texts, val_texts, train_labels, val_labels = train_test_split(
    #    df['Merchant'], df['label'], test_size=0.2, random_state=42
    # )
    train_texts = df['Merchant']
    train_labels = df['label']

    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(Constants.bert_pre_trained_model_name)

    # Hyperparameters

    # Hyperparameters
    MAX_LEN = 128  # Max length for BERT input sequences
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    # Create datasets and data loaders
    train_dataset = ExpenseDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, MAX_LEN)
    #  val_dataset = ExpenseDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Load pre-trained BERT model for classification
    num_labels = len(df['ExpenseType'].unique())
    model = BertForSequenceClassification.from_pretrained(Constants.bert_pre_trained_model_name, num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f'Train loss: {train_loss}, accuracy: {train_acc}')
    #   val_acc, val_loss = eval_model(model, val_loader, device)
    #   print(f'Validation loss: {val_loss}, accuracy: {val_acc}')

    # Save the model and label encoder in the trained_model folder
    model_dir = "trained_model"
    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(os.path.join(model_dir, "bert_expense_classification_model"))
    torch.save(label_encoder, os.path.join(model_dir, "expense_label_encoder.pth"))

    return "Model trained and saved successfully"


# Initialize the Flask app for training service
app = Flask(__name__)


# Define the training endpoint
@app.route('/train', methods=['POST'])
def train():
    """
    adding the code to receive the file as intput parameter in post body, and save it

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)
    """

    try:
        file_path = Constants.file_path
        response = train_model(file_path)
        print("message : " + response)
    except Exception as e:
        print(f"Error : {e}")


# Start the Flask app
if __name__ == '__main__':
    # app.run(debug=True, host='127.0.0.1', port=5001)
    train()
