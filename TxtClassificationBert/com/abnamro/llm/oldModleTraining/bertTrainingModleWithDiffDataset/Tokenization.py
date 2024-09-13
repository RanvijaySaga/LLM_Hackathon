from transformers import BertTokenizer


def get_tokenizer():
    return BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased', clean_up_tokenization_spaces=True)
