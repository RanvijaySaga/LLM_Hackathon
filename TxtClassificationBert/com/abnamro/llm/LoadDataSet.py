import pandas as pd

from sklearn.preprocessing import LabelEncoder


def load_data_set(data_file):
    data_frame = pd.read_csv(data_file)
    texts = data_frame['review'].tolist()
    labels = [1 if sentiment == "positive" else 0 for sentiment in data_frame['sentiment'].tolist()]
    return texts, labels


def convert_category_to_numeric(data_file):
    data_frame = pd.read_csv(data_file)
    label_encoder = LabelEncoder()
    data_frame['label'] = label_encoder.fit_transform(data_frame['category'])
    print(data_frame)

    return label_encoder, data_frame
