import os
import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification


# Load the pre-trained model and tokenizer from local storage
def load_model_and_tokenizer(model_path, label_encoder_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    label_encoder = torch.load(label_encoder_path)
    return model, tokenizer, label_encoder


# Function to make predictions
def predict_expense_type(description, model, tokenizer, label_encoder, max_len=128):
    inputs = tokenizer.encode_plus(
        description,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        predicted_label_idx = torch.argmax(logits, dim=1).item()

    predicted_expense_type = label_encoder.inverse_transform([predicted_label_idx])[0]
    return predicted_expense_type


# Initialize the Flask app for prediction service
app = Flask(__name__)


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'description' not in data:
        return jsonify({"error": "No description provided"}), 400

    description = data['description']

    try:
        # Load the model and tokenizer
        model_path = os.path.join("trained_model", "bert_expense_classification_model")
        label_encoder_path = os.path.join("trained_model", "expense_label_encoder.pth")
        model, tokenizer, label_encoder = load_model_and_tokenizer(model_path, label_encoder_path)

        # Predict the ExpenseType using the model
        predicted_expense_type = predict_expense_type(description, model, tokenizer, label_encoder)
        return jsonify({"description": description, "predicted_expense_type": predicted_expense_type}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.2', port=5002)
