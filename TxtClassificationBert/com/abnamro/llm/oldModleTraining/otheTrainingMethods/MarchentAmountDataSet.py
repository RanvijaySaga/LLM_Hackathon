import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 1. Custom Dataset Class for Loading Data
class MerchantAmountDataset(Dataset):
    def __init__(self, merchants, amounts, expense_types, tokenizer, max_len):
        self.merchants = merchants
        self.amounts = amounts
        self.expense_types = expense_types
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.merchants)

    def __getitem__(self, index):
        merchant = str(self.merchants[index])
        amount = str(self.amounts[index])
        expense_type = self.expense_types[index]

        # Combine merchant and amount as input text
        input_text = f"{merchant} {amount}"

        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'merchant_amount': input_text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(expense_type, dtype=torch.long)
        }

# 2. Function to Train the Model
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        predictions = outputs.logits.argmax(dim=1)
        correct_predictions += torch.sum(predictions == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(data_loader)

# 3. Function for Evaluation
def eval_model(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    losses = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            predictions = outputs.logits.argmax(dim=1)
            correct_predictions += torch.sum(predictions == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(data_loader)

# 4. Main Function to Run Text Classification
def main():
    # Load dataset (assumed to have 'Merchant', 'Amount', and 'ExpenseType' columns)
    df = pd.read_csv('/resources/TrainingData.csv')

    # Convert ExpenseType to numeric labels using LabelEncoder
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['ExpenseType'])

    # Train-test split
    train_merchants, val_merchants, train_amounts, val_amounts, train_labels, val_labels = train_test_split(
        df['Merchant'], df['Amount'], df['label'], test_size=0.2, random_state=42
    )

    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Hyperparameters
    MAX_LEN = 64  # Adjust depending on the length of Merchant names and amounts
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 2e-5  # Learning rate

    # Create DataLoaders
    train_dataset = MerchantAmountDataset(train_merchants.tolist(), train_amounts.tolist(), train_labels.tolist(), tokenizer, MAX_LEN)
    val_dataset = MerchantAmountDataset(val_merchants.tolist(), val_amounts.tolist(), val_labels.tolist(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Load pre-trained BERT model
    num_labels = len(df['ExpenseType'].unique())  # Number of unique expense types
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps // 10, gamma=0.1)

    # Training Loop
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, val_loader, device)
        print(f'Validation loss {val_loss} accuracy {val_acc}')

    # Save the model and label encoder for inference
    model.save_pretrained("bert_merchant_amount_classification_model")
    torch.save(label_encoder, "label_encoder.pth")

    # Print example results
    print("Model training complete. Example predictions:")

    # Make predictions on the validation set

    model.eval()
    for i in range(5):  # Show 5 example predictions
        merchant = val_merchants.iloc[i]
        amount = val_amounts.iloc[i]
        label = val_labels.iloc[i]
        input_text = f"{merchant} {amount}"
        inputs = tokenizer.encode_plus(input_text, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding='max_length').to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_label = torch.argmax(logits, dim=1).item()
        print(f"Merchant: {merchant}, Amount: {amount}, True Label: {label_encoder.inverse_transform([label])[0]}, Predicted: {label_encoder.inverse_transform([predicted_label])[0]}")

# Run the script
if __name__ == "__main__":
    main()
