# -*- coding: utf-8 -*-
# @Time    : 2021/6/9 12:51 AM
# @Author  : tonysu
# @File    : ffm.py
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
K = tf.keras.backend


class CrossLayer(keras.layers.Layer):
    def __init__(self, field_dict, field_dim, input_dim, output_dim=30, **kwargs):
        self.field_dict = field_dict
        self.field_dim = field_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.field_cross = K.variable(0, dtype='float32')
        self.field_cross_origin = K.variable(0, dtype='float32')
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.field_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(CrossLayer, self).build(input_shape)

    def call(self, x):
        self.field_cross = self.field_cross_origin
        for i in range(self.input_dim):
            for j in range(i+1, self.input_dim):
                weight = tf.math.reduce_sum(tf.math.multiply(self.kernel[i, self.field_dict[j]], self.kernel[j, self.field_dict[i]]))
                value = tf.math.multiply(weight, tf.math.multiply(x[:,i], x[:,j]))
                self.field_cross = tf.math.add(self.field_cross, value)
        return self.field_cross

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


def FFM(feature_dim, field_dict, field_dim, output_dim=30):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(1)(inputs)
    cross = CrossLayer(field_dict, field_dim, feature_dim, output_dim)(inputs)
    cross = tf.keras.layers.Reshape((1,))(cross)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['binary_accuracy'])
    return model


def train():
    field_dict = {i:i//5 for i in range(30)}
    ffm = FFM(30, field_dict, 6, 30)
    ffm.summary()
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                        random_state=27, stratify=data.target)
    ffm.fit(X_train, y_train, epochs=3, batch_size=16, validation_data=(X_test, y_test))
    return ffm


if __name__ == '__main__':
    ffm = train()