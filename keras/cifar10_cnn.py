import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout

'''
[Keras] CNNでCifar10の画像認識を行う
'''

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0",
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

# データ取得
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 正規化
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

# One-Hot表現
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# モデルの構築
# 入力
inputs = Input(shape=(32, 32, 3))

# 畳み込み1
x = Conv2D(32,
           (3, 3),
           padding='same',
           activation='relu')(inputs)
# プーリング1
x = MaxPooling2D((2, 2), padding='same')(x)
# Dropout
x = Dropout(0.2)(x)

# 畳み込み2
x = Conv2D(64,
           (3, 3),
           padding='same',
           activation='relu')(x)
# プーリング2
x = MaxPooling2D((2, 2), padding='same')(x)
# Dropout
x = Dropout(0.3)(x)

# 3次元->1次元に変換
x = Flatten()(x)

# 全層結合1
x = Dense(512, activation='relu')(x)
# Dropout
x = Dropout(0.5)(x)
# 全層結合2 最終層の出力はクラス数の10
outputs = Dense(10, activation='softmax')(x)

# Model生成
model = Model(inputs=inputs, outputs=outputs)
# Modelコンパイル
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 学習開始
history = model.fit(x_train,
                    y_train,
                    batch_size=100,
                    epochs=20,
                    validation_data=(x_test, y_test))
model.summary()