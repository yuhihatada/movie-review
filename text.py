# 2項分類
# 映画のレビューが肯定的か否定的かの判別
# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras
import numpy as np

# データのインポート
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# データの観察
# print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# print(train_data[0]) # テキストが整数化されて配列要素それぞれ（[i]）に入っている二次元データになっている

# 単語を整数にマッピングする辞書のダウンロード
word_index = imdb.get_word_index()

# インデックスの最初の方は予約済み
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 辞書関数の定義
def decode_review(text):
  return ' '.join([reverse_word_index.get(i, '?') for i in text])

# 辞書関数を用いた整数羅列の変換
# print(decode_review(train_data[0])) # 1整数1単語に対応して変換される

# 映画レビューの長さを標準化
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# 入力の形式は映画レビューで使われている語彙数（10,000語）
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# 損失関数とクエリ最適化の設定
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 検証用データの作成
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# モデルの訓練
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# モデルの評価
results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)

# 訓練中に発生した全てのことを記録
history_dict = history.history
history_dict.keys()


# 訓練時と検証時の損失と正解率を比較するグラフを表示する
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # 図のクリア

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

