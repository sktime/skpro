import edward as ed
import numpy as np
import tensorflow as tf

from ..base import BayesianVendorEstimator, BayesianVendorInterface


class Edwardlib(BayesianVendorEstimator):

    pass


class EdwardlibInterface(BayesianVendorInterface):

    def __init__(self, sample_size=500):
        self.sample_size = sample_size
        self.X_ = None

    def on_fit(self, X, y):
        self.X_ = tf.placeholder(tf.float32, X.shape)
        w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
        b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
        y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))

        qw = Normal(loc=tf.Variable(tf.random_normal([D])),
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
        qb = Normal(loc=tf.Variable(tf.random_normal([1])),
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

        inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
        inference.run(n_samples=5, n_iter=250)



    def on_predict(self, X):
        y_post = ed.copy(y, {w: qw, b: qb})


    def samples(self):
        return self.ppc_['y_pred'].T


