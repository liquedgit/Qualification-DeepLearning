import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import time

data_1 = pd.read_csv('./NLP/train.csv')
data_2 = pd.read_csv('./NLP/test.csv')
data_3 = pd.read_csv('./NLP/val.csv')
data1_sentences = data_1['Text']
data2_sentences = data_2['Text']
data3_sentences = data_3['Text']
sentences = data1_sentences.tolist() + data2_sentences.tolist() + data3_sentences.tolist()

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0],True)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
input_sequences = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]

max_length = max([len(seq) for seq in input_sequences])

x = [seq[:-1] for seq in input_sequences]
y = [seq[1:] for seq in input_sequences]

x = pad_sequences(x, maxlen=max_length-1, padding='post')
y = pad_sequences(y, maxlen=max_length-1, padding='post')


x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=int(time.time()))
x_val, x_test, y_val, y_text = train_test_split(x, y, test_size=0.5, random_state=int(time.time()))

def dot_product_scaled(query, key, value, mask):
    matmul = tf.matmul(query, key, transpose_b=True)
    d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul / tf.math.sqrt(d_k)

    if mask is not None:
        logits = matmul / tf.math.sqrt(d_k)
    
    weights = tf.nn.softmax(logits, axis=-1)
    out = tf.matmul(weights, value)

    return out, weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.d_model = d_model
    
        assert d_model % heads == 0
        
        self.depth = d_model // heads

        self.v = tf.keras.layers.Dense(d_model)
        self.k = tf.keras.layers.Dense(d_model)
        self.q = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split(self, x, batch):
        x = tf.reshape(x, (batch, -1, self.heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call (self, query, key, value, mask):
        batch = tf.shape(query)[0]
        query = self.split(self.q(query), batch)
        key = self.split(self.k(key), batch)
        value = self.split(self.v(value), batch)

        scaled, weights = dot_product_scaled(query, key, value, mask)
        scaled = tf.transpose(scaled, perm=[0,2,1,3])
        concat = tf.reshape(scaled, (batch, -1, self.d_model))
        output = self.dense(concat)

        return output, weights

def feed_fw(d_mode, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

def encoding(position, d_model):
    angle = 1/np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle[:, 0::2] = np.sin(position * angle[:, 0::2])
    angle[:, 1::2] = np.sin(position * angle[:, 1::2])
    pos_encoding = angle[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype = tf.float32)

num_layers = 2
d_model = 64
heads = 8
dff = 256
input_vocab_size = len(tokenizer.vocab)
maximum_position_encoding = max_length
dropout_rate = 0.5

inputs = tf.keras.layers.Input(shape=(max_length-1,))
x = tf.keras.layers.Embedding(input_vocab_size, d_model)(inputs)
x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
x += encoding(maximum_position_encoding, d_model)
 
x = tf.keras.layers.Dropout(dropout_rate)(x)

for _ in range(num_layers):
    attn_output, _ = MultiHeadAttention(d_model, heads)(x, x, x, None)
    x = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(x + attn_output)
    ff_output = feed_fw(d_model, dff)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

outputs = tf.keras.layers.Dense(input_vocab_size, activation='softmax')(x)
tf = tf.keras.Model(inputs=inputs, outputs=outputs)

tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tf.summary()

history = tf.fit(
    x_train, np.expand_dims(y_train, -1),
    validation_data=(x_val, np.expand_dims(y_val, -1)),
    epochs=10, batch_size=64
)

tf.save("best_model.h5")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')

plt.savefig("fig.png")
plt.show()
