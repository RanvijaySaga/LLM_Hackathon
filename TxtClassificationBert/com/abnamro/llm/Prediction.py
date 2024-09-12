import torch
from com.abnamro.llm.Tokenization import get_tokenizer
from com.abnamro.llm.TrainModel import train_model



def predict(texts, count):
    global model, label_encoder

    # try to use same model again
    # this portion needs to be replaced to  pick the previously trained model and add more behaviour data and not to replace it.

    if count < 1:
        model, label_encoder = train_model()
        count = 1

    # Tokenize and encode
    tokenizer = get_tokenizer()
    input_data = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**input_data)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return [label_encoder.inverse_transform([pred])[0] for pred in predictions]


def main():
    # use the created model to predict next behaviour
    print(predict(["IKEA Amsterdam"],0))
    print(predict(["C&A"],1))
    print(predict("JUMBO",1))
    print(predict(["Acti on"],1))


if __name__ == "__main__":
    main()
