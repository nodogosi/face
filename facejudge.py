# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 01:17:50 2021

@author: yo_wa
"""
import numpy as np
# 必要なモジュールの読み込み
from PIL import Image
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# モデルの読込
model = load_model('./my_model.h5')

image = Image.open('./men_pic/0c42f7a0.jpg')
_imga = np.array(image, dtype=np.int32)
Y_pred = model.predict(_imga.reshape(1,100,100,3))
# 二値分類は予測結果の確率が0.5以下なら0,
# それより大きければ1となる計算で求める
Y_pred_cls = (Y_pred > 0.5).astype("int32")
print('女性っぽさ:',Y_pred)
plt.imshow(_imga)
plt.show()