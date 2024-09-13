model_path = "./model/bert_expense_classification_model"
label_encoder_path = "expense_label_encoder.pth"
file_path = "E:/LLM/pycharm/LLM_Hackathon/TxtClassificationBert/resources/TrainingData.csv"

bert_pre_trained_model_name = 'GroNLP/bert-base-dutch-cased'
# bert_pre_trained_model_name = 'bert-base-uncased'

EXPENSE_TYPE_CATEGORY_DICT = {
    "Dining": "O",
    "Clothing": "O",
    "Entertainment": "O",
    "Groceries": "M",
    "Others": "M",
    "Utilities": "M",
    "Healthcare": "M",
    "Fee": "M",
    "Fitness": "M",
    "Income": "M"
}