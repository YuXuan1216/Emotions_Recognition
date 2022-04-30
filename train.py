# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:24:56 2022

@author: aiilab
"""

# IMPORT MODULES
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def show_train_history(train_history):
    plt.figure(figsize=(10,5))
    plt.plot(train_history.history['accuracy'])
    plt.plot(train_history.history['val_accuracy'])
    plt.xticks([i for i in range(len(train_history.history['accuracy']))])
    plt.title('Train History')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.xticks([i for i in range(len(train_history.history['loss']))])
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    

# -----------------------------1.設置模型架構--------------------------------
# 載入keras模型(更換輸出圖片尺寸)
model = InceptionResNetV2(include_top=False, weights='imagenet',input_tensor=Input(shape=(80, 80, 3)))

# # 定義輸出層
x = model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)

# 編譯模型
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------2.設置callbacks----------------------------
# 設定earlystop條件
estop = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

# 設定模型儲存條件
checkpoint = ModelCheckpoint('InceptionResNetV2_checkpoint_v2.h5', verbose=1,
                          monitor='val_loss', save_best_only=True,
                          mode='min')

# 設定lr降低條件
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                           patience=5, mode='min', verbose=1,
                           min_lr=0.0001)

# -----------------------------3.設置資料集--------------------------------
# 設定ImageDataGenerator參數(路徑、批量、圖片尺寸)
train_dir = './train/'
valid_dir = './valid/'
test_dir = './test/'
batch_size = 64
target_size = (80, 80)

# # 設定批量生成器
train_datagen = ImageDataGenerator(rescale=1./255, 
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2, 
                                    fill_mode="nearest")
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# # 讀取資料集+批量生成器，產生每epoch訓練樣本
train_generator = train_datagen.flow_from_directory(train_dir,
                                      target_size=target_size,
                                      batch_size=batch_size)

valid_generator = val_datagen.flow_from_directory(valid_dir,
                                      target_size=target_size,
                                      batch_size=batch_size)

test_generator = test_datagen.flow_from_directory(test_dir,
                                      target_size=target_size,
                                      batch_size=batch_size,
                                      shuffle=False)

# -----------------------------4.訓練模型------------------------------
# 重新訓練權重
history = model.fit_generator(train_generator,
                   epochs = 50, verbose = 1,
                   steps_per_epoch = train_generator.samples//batch_size,
                   validation_data = valid_generator,
                   validation_steps = valid_generator.samples//batch_size,
                   callbacks=[checkpoint, estop, reduce_lr])

# -----------------------5.儲存模型、紀錄學習歷程------------------------
# 儲存模型
model.save('./InceptionResNetV2_retrained_v2.h5')
print('已儲存InceptionResNetV2_retrained_v2.h5')

show_train_history(history)

# -------------------------6.模型準確度--------------------------
print(train_generator.class_indices)

train_loss, train_acc = model.evaluate_generator(train_generator,steps=train_generator.samples//batch_size,verbose=1)
print('train_acc:', train_acc)
print('train_loss:', train_loss)

valid_loss, valid_acc = model.evaluate_generator(valid_generator,steps=valid_generator.samples//batch_size,verbose=1)
print('valid acc:', valid_acc)
print('valid loss:', valid_loss)

test_loss, test_acc = model.evaluate_generator(test_generator,steps=test_generator.samples//batch_size,verbose=1)
print('test acc:', test_acc)
print('test loss:', test_loss)
