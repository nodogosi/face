# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 23:49:03 2021

@author: yo_wa
"""

#やりたいこと
#100×100のjpeg画像から、男女を判別したい。
#
#実現するための手順
#1.100×100の男女の顔のjpeg画像を配列に変換する
#2.CNNを構築し、①で作成した顔画像データを学習させる。

import numpy as np
# 必要なモジュールの読み込み
from PIL import Image
import glob
import matplotlib.pyplot as plt
# Keras
from tensorflow import keras
# ライブラリのインポート
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

# データの分割
from sklearn.model_selection import train_test_split

#****************************************************************************#
#1.100×100の男女の顔のjpeg画像を配列に変換する
#****************************************************************************#
# 指定したディレクトリから、拡張子がjpgであるもののファイル名リストを取得する
imglist = glob.glob('./men_pic/*.jpg')
# ファイル名リストから、ファイル名imgを一つずつ取り出す。これをリストが終わるまで繰り返す
for i,img in enumerate(imglist):
# ファイル名imgを読み出す
 image = Image.open(img)
# 画像データ「image」（3次元、縦100px、横100px、RGB値 ）を配列にし、
# 4次元の説明変数「X_men」に追加する（3次元の画像データをリスト化することで4次元になる）。
# 画像データ「image」を4次元の説明変数「X_men」に追加するために、reshapeで4次元にしている。
 _imga = np.array(image, dtype=np.float64).reshape(1,100,100,3)
# 説明変数「X_men」は初回は存在しないので、初回だけは画像データ「image」を代入する
 if i == 0:
  X_men = _imga
# 初回以降は、説明変数「X_men」に追加していく
 else:
  X_men = np.append(X_men,_imga,axis=0)
 
# 既存の画像を右回転させた男性画像を学習させたい
# ファイル名リストから、ファイル名imgを一つずつ取り出す。これをリストが終わるまで繰り返す
for img in imglist:
# ファイル名imgを読み出す
 image = Image.open(img)
# 既存の画像を右回転させる。回転角度は乱数を用いる。最大45度
 im_rotate = image.rotate(np.random.rand()*45)
# 画像データ「image」（3次元、縦100px、横100px、RGB値 ）を配列にし、
# 4次元の説明変数「X_men」に追加する（3次元の画像データをリスト化することで4次元になる）。
# 画像データ「image」を4次元の説明変数「X_men」に追加するために、reshapeで4次元にしている。
 _imga = np.array(image, dtype=np.float64).reshape(1,100,100,3)
 X_men = np.append(X_men,_imga,axis=0)

# 既存の画像を左回転させた男性画像を学習させたい
# 処理は右回転とほぼ同じ。変化点は左回転させるために-45を乗算している箇所
for img in imglist:
 image = Image.open(img)
 im_rotate = image.rotate(np.random.rand()*(-45))
 _imga = np.array(image, dtype=np.float64).reshape(1,100,100,3)
 X_men = np.append(X_men,_imga,axis=0)

# 説明変数の数量分、目的変数をつくる(目的変数は「女性らしさ」を示す。つまり男は0)
Y_men = np.stack([0 for j in range(0, len(X_men))])
print('男性画像',X_men.shape,Y_men.shape)

# 画像データの読込（女性画像）
# 方針は男性画像と同じ。右回転と左回転の画像データを、説明変数「X_women」に追加していく
imglist = glob.glob('./women_pic/*.jpg')
for i,img in enumerate(imglist):
 image = Image.open(img)
 _imga = np.array(image, dtype=np.float64).reshape(1,100,100,3)
 if i == 0:
  X_women = _imga
 else:
  X_women = np.append(X_women,_imga,axis=0)

#回転させた女性画像を追加する
for img in imglist:
 image = Image.open(img)
 im_rotate = image.rotate(np.random.rand()*45)
 _imga = np.array(image, dtype=np.float64).reshape(1,100,100,3)
 X_women = np.append(X_women,_imga,axis=0)
 
#回転させた女性画像を追加する
for img in imglist:
 image = Image.open(img)
 im_rotate = image.rotate(np.random.rand()*(-45))
 _imga = np.array(image, dtype=np.float64).reshape(1,100,100,3)
 X_women = np.append(X_women,_imga,axis=0) 

# 説明変数の数量分、目的変数をつくる(目的変数は「女性らしさ」を示す。つまり女性は1)
Y_women = np.stack([1 for j in range(0, len(X_women))])
print('女性画像',X_women.shape,Y_women.shape)

# 男性の説明変数と、女性の説明変数を合体させる
X = np.concatenate([X_men,X_women])
# 男性の目的変数と、女性の目的変数を合体させる
Y = np.concatenate([Y_men,Y_women])

print(X.shape)
print(Y.shape)

#****************************************************************************#
#2.CNNを構築し、1で作成した顔画像データを学習させる。
#****************************************************************************#


# 訓練データとテストデータの分割
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
# 訓練データから検証データを分割
X_train,X_valid,Y_train,Y_valid = train_test_split(X_train,Y_train,test_size=0.2,random_state=0)
print(X_train.shape,X_valid.shape,X_test.shape)
print(Y_train.shape,Y_valid.shape,Y_test.shape)

# モデルの初期化
model = keras.Sequential()

#１層目
model.add(Conv2D(64,kernel_size=3,padding='same',strides=1,input_shape=(100,100,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#繰り返し
model.add(Conv2D(128,kernel_size=3,padding='same',strides=1,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#一次元に変換
model.add(Flatten())
model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
#出力層（二値分類の出力なので活性化関数はシグモイド）
model.add(Dense(1,activation='sigmoid'))

#モデル構築（二値分類なので目的関数はbinary_crossentropy）
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

# モデルの構造を表示
model.summary()

# 学習を実施し、結果をlogで受け取る。エポック数はとりあえず10(1000くらいまで上げたほうがいいがかなり時間がかかる、サンプルでは早く終わるように少な目で設定)
# EarlyStoppingを使用して過学習防止する
log=model.fit(X_train,Y_train,epochs=10,batch_size=32,verbose=True,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=100,verbose=1)],
             validation_data=(X_valid,Y_valid))

#エポック数の増加と、目的関数の出力の変化をプロット
#（エポック数の増加により学習が進んで小さくなるが、増加しはじめたら過学習）
plt.plot(log.history['loss'],label='loss')
plt.plot(log.history['val_loss'],label='val_loss')
plt.ylim(0, 1)
plt.legend(frameon=False)
plt.xlabel('epochs')
plt.ylabel('binary_crossentropy')
plt.show()

Y_pred = model.predict(X_test)
# 二値分類は予測結果の確率が0.5以下なら0,
# それより大きければ1となる計算で求める
Y_pred_cls = (Y_pred > 0.5).astype("int32")

# classification_reportを使い、モデルの評価を実施
from sklearn.metrics import classification_report

print(classification_report(Y_pred_cls,Y_test))

# モデルの保存（HDF5形式）
model.save('./my_model.h5')

