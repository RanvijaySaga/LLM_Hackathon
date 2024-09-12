from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments

from com.abnamro.llm.LoadDataSet import convert_category_to_numeric
from com.abnamro.llm.Tokenization import get_tokenizer
from com.abnamro.llm.TextClassificationDataset import TextClassificationDataset

data_file = "E:/LLM/pycharm/TxtClassificationBert/resources/transactions.csv"
bert_classifier_name = 'GroNLP/bert-base-dutch-cased'


def get_training_args():
    return TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )


def get_trainer(model, dataset):
    return Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=dataset
    )


def train_model():
    label_encoder, data_frame = convert_category_to_numeric(data_file)

    # tokenizer
    tokenizer = get_tokenizer()
    tokenized_data = tokenizer(data_frame['description'].tolist(), padding=True, truncation=True, return_tensors="pt")

    # Create dataset and dataloader
    dataset = TextClassificationDataset(tokenized_data, data_frame['label'].tolist())
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Load pre-trained BERT model for classification
    model = BertForSequenceClassification.from_pretrained(bert_classifier_name, num_labels=len(label_encoder.classes_))

    trainer = get_trainer(model, dataset)
    trainer.train()

    return model, label_encoder
