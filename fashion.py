# 分類問題
# 画像の判別
# keras.layersのクラスに用意されているメソッド
# ↓
# Conv2D : 2次元のConvolution層（畳み込み層）
# MaxPooling2D : 2次元のPooling層(Max Pooling)
# Activation : 活性化関数の層。reluとかsigmoidとかsoftmaxとかを指定できます。
# Dropout : ドロップアウト層。挿入すると、自動でドロップアウトしてくれます。
# Flatten : 多次元の配列を1次元の配列に変換します。
# Dense : 全結合層

# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

# 60,000枚の画像を訓練に、10,000枚の画像を、ネットワークが学習した画像分類の正確性を評価するのに使います。TensorFlowを使うと、下記のようにFashion MNISTのデータを簡単にインポートし、ロードすることが出来ます。
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 画像はそれぞれ単一のラベルに分類されます。データセットには下記のクラス名が含まれていないため、後ほど画像を出力するときのために、クラス名を保存しておきます。
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# データの観察
# train_images.shape
# len(train_labels)
# train_labels
# test_images.shape
# len(test_labels)

# ネットワークを訓練する前に、データを前処理する必要があります。最初の画像を調べてみればわかるように、ピクセルの値は0から255の間の数値です。
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
# plt.show()

# ニューラルネットワークにデータを投入する前に、これらの値を0から1までの範囲にスケールします。そのためには、画素の値を255で割ります。
# 訓練用データセットとテスト用データセットは、同じように前処理することが重要です。
train_images = train_images / 255.0
test_images = test_images / 255.0

# 訓練用データセットの最初の25枚の画像を、クラス名付きで表示してみましょう。ネットワークを構築・訓練する前に、データが正しいフォーマットになっていることを確認します。
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 層の設定
# ニューラルネットワークを形作る基本的な構成要素は層（layer）です。層は、入力されたデータから「表現」を抽出します。それらの「表現」は、今取り組もうとしている問題に対して、より「意味のある」ものであることが期待されます。
# ディープラーニングモデルのほとんどは、単純な層の積み重ねで構成されています。tf.keras.layers.Dense のような層のほとんどには、訓練中に学習されるパラメータが存在します。
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

# モデルが訓練できるようになるには、いくつかの設定を追加する必要があります。それらの設定は、モデルのコンパイル(compile）時に追加されます。
# 損失関数（loss function） —訓練中のモデルが不正確であるほど大きな値となる関数です。この関数の値を最小化することにより、訓練中のモデルを正しい方向に向かわせようというわけです。
# オプティマイザ（optimizer）—モデルが見ているデータと、損失関数の値から、どのようにモデルを更新するかを決定します。
# メトリクス（metrics） —訓練とテストのステップを監視するのに使用します。下記の例ではaccuracy （正解率）、つまり、画像が正しく分類された比率を使用しています。
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練を開始するには、model.fit メソッドを呼び出します。モデルを訓練用データに "fit"（適合）させるという意味です。
model.fit(train_images, train_labels, epochs=5)

# テスト用データセットに対するモデルの性能を比較します。
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 画像の分類の予測
predictions = model.predict(test_images)

# モデルがテスト用データセットの画像のひとつひとつを分類予測した結果です。最初の予測を見てみましょう。
predictions[0]

# 予測結果は、10個の数字の配列です。これは、その画像が10の衣料品の種類のそれぞれに該当するかの「確信度」を表しています。どのラベルが一番確信度が高いかを見てみましょう。
np.argmax(predictions[0])

# というわけで、このモデルは、この画像が、アンクルブーツ、class_names[9] である可能性が最も高いと判断したことになります。これが正しいかどうか、テスト用ラベルを見てみましょう。
test_labels[0]

# 10チャンネルすべてをグラフ化してみることができます。
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
      color = 'blue'
  else:
      color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# 0番目の画像と、予測、予測配列を見てみましょう。
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
# plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
# plt.show()

# 予測の中のいくつかの画像を、予測値とともに表示してみましょう。正しい予測は青で、誤っている予測は赤でラベルを表示します。
# 数字は予測したラベルのパーセント（100分率）を示します。自信があるように見えても間違っていることがあることに注意してください。
# X個のテスト画像、予測されたラベル、正解ラベルを表示します。
# 正しい予測は青で、間違った予測は赤で表示しています。
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
# plt.show()

# 最後に、訓練済みモデルを使って1枚の画像に対する予測を行います。
# テスト用データセットから画像を1枚取り出す
img = test_images[0]
print(img.shape)

# tf.keras モデルは、サンプルの中のバッチ（batch）あるいは「集まり」について予測を行うように作られています。そのため、1枚の画像を使う場合でも、リスト化する必要があります。
# 画像を1枚だけのバッチのメンバーにする
img = (np.expand_dims(img,0))
print(img.shape)

# そして、予測を行います。
predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

# model.predict メソッドの戻り値は、リストのリストです。リストの要素のそれぞれが、バッチの中の画像に対応します。バッチの中から、（といってもバッチの中身は１つだけですが）予測を取り出します。
res = (np.argmax(predictions_single[0]))
print(class_names[res])
