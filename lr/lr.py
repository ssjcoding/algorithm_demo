# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 10:41 PM
# @Author  : tonysu
# @File    : lr.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import tensorflow.keras as keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def lr_model():
    inputs = Input((30, ))
    pred = Dense(units=1,
                 bias_regularizer=keras.regularizers.l2(0.01),
                 kernel_regularizer=keras.regularizers.l1(0.02),
                 activation=tf.nn.sigmoid)(inputs)
    lr = keras.Model(inputs, pred)
    lr.compile(loss='binary_crossentropy',
               optimizer=keras.optimizers.Adam(),
               metrics=['binary_accuracy'])
    return lr


def train():
    lr = lr_model()
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                        random_state=27, stratify=data.target)

    lr.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_test, y_test))


if __name__ == '__main__':
    lr = train()

