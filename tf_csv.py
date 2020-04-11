import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import pandas as pd
import numpy as np
import functools


TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_csv = keras.utils.get_file('train_csv',TRAIN_DATA_URL)
test_csv = keras.utils.get_file('test_csv',TEST_DATA_URL)

train_df = pd.read_csv(TRAIN_DATA_URL)

batch_size = 32
label_column = 'survived'
labels = [0, 1]
def get_dataset(file_csv_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_csv_path,
        batch_size = batch_size,
        label_name = label_column,
        num_epochs=1,
        ignore_errors=True)
    return dataset

feature_columns = []
CATEGORIES = {
    'sex': train_df.sex.unique(),
    'class': train_df['class'].unique(),
    'deck': train_df.deck.unique(),
    'embark_town': train_df.embark_town.unique(),
    'alone': train_df.alone.unique()
}
categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

def process_continuous_data(mean, data):
    # 标准化数据
    data = tf.cast(data, tf.float32) * 1/(2*mean)
    return tf.reshape(data, [-1, 1])

MEANS = {
    'age' : train_df.age.mean(),
    'n_siblings_spouses' : train_df.n_siblings_spouses.mean(),
    'parch' : train_df.parch.mean(),
    'fare' : train_df.fare.mean(),
} 
numerical_columns = []

for feature in MEANS.keys():
    num_col = tf.feature_column.numeric_column(
        feature, normalizer_fn=functools.partial(
            process_continuous_data, MEANS[feature]))
    numerical_columns.append(num_col)

def make_model():
    preprocessing_layer = tf.keras.layers.DenseFeatures(
        categorical_columns+numerical_columns)
    model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])


    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


train_ds = get_dataset(train_csv).shuffle(1000)
test_ds = get_dataset(test_csv)

model = make_model()
history = model.fit(train_ds, epochs=20)

test_loss, test_accuracy = model.evaluate(test_ds)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict_classes(test_ds)

# 显示部分结果
for prediction, survived in zip(predictions[:20], list(test_ds)[0][1][:20]):
  print("Predicted: {}".format("SURVIVED" if bool(prediction[0]) else "DIED    "),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))
