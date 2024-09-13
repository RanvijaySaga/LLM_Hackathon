import pandas as pd

from sklearn.preprocessing import LabelEncoder


def convert_category_to_numeric(data_file):
    data_frame = pd.read_csv(data_file)
    label_encoder = LabelEncoder()
    data_frame['label'] = label_encoder.fit_transform(data_frame['ExpenseType'])
    print(data_frame)

    return label_encoder, data_frame
