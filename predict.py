# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 18:26:56 2022

@author: aiilab
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os, cv2
import matplotlib.pyplot as plt


happy_dir = r"./test/happy"
angry_dir = r"./test/angry"
normal_dir = r"./test/normal"
sad_dir = r"./test/sad"
dic = {0:'angry', 1:'happy', 2:'normal', 3:'sad'}
# # 關閉GPU加速功能(建議安裝無GPU版本，縮短初始化時間)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# 載入模型
try:
    model = load_model('./InceptionResNetV2_retrained_v2.h5')
    print("success")
except:
    print("error")
    

dir_=[angry_dir, happy_dir, normal_dir, sad_dir]

for d in range(len(dir_)):
    for i in os.listdir(dir_[d]):
        img = cv2.imread(dir_[d]+'/'+i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (80,80))
        
        img_tmp = img
        
        # 樣本預測
        img = np.expand_dims(img, axis=0)/255.0
        prediction=model.predict(img)
        index = np.argmax(prediction)
        prediction = dic[index]
        print(prediction) # 預測結果
        
        plt.title(prediction)
        plt.imshow(img_tmp)
        plt.show()

# 設定ImageDataGenerator參數(路徑、批量、圖片尺寸)
test_dir = './test/'
batch_size = 64
target_size = (80, 80)

# # 設定批量生成器
test_datagen = ImageDataGenerator(rescale=1./255)

# # 讀取資料集+批量生成器，產生每epoch訓練樣本
test_generator = test_datagen.flow_from_directory(test_dir,
                                      target_size=target_size,
                                      batch_size=batch_size,
                                      shuffle=False)

test_loss, test_acc = model.evaluate_generator(test_generator,steps=test_generator.samples//batch_size,verbose=1)
print('test acc:', test_acc)
print('test loss:', test_loss)