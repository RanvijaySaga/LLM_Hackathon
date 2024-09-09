import torch
from com.abnamro.llm.Tokenization import get_tokenizer
from com.abnamro.llm.TrainModel import train_model


def predict(texts):
    global model, label_encoder
    count = 0

    if count < 1:
        model, label_encoder = train_model()

    # Tokenize and encode
    tokenizer = get_tokenizer()
    input_data = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**input_data)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return [label_encoder.inverse_transform([pred])[0] for pred in predictions]


# Example usage
print(predict(["HBO"]))
print(predict(["Jumbo"]))
print(predict(["ZARA"]))
