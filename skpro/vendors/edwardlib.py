import numpy as np

import tensorflow as tf
import edward as ed
from edward.models import Normal

from ..base import BayesianVendorEstimator, BayesianVendorInterface


class Edwardlib(BayesianVendorEstimator):

    pass


class EdwardlibInterface(BayesianVendorInterface):

    def __init__(self, sample_size=500):
        self.sample_size = sample_size
        self.X_ = None
        self.y_ = None
        self.y_pred = None

    def on_fit(self, X, y):
        # define shared variable
        self.X_ = tf.placeholder(tf.float32, X.shape)

        w = Normal(loc=tf.zeros(X.shape[1]), scale=tf.ones(X.shape[1]))
        b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
        self.y_ = Normal(loc=ed.dot(self.X_, w) + b, scale=tf.ones(X.shape[0]))

        # inference
        qw = Normal(loc=tf.Variable(tf.random_normal([X.shape[1]])),
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([X.shape[1]]))))
        qb = Normal(loc=tf.Variable(tf.random_normal([1])),
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

        inference = ed.KLqp({w: qw, b: qb}, data={self.X_: X, self.y_: y})
        inference.run(n_samples=5, n_iter=250)

        self.y_pred = ed.copy(self.y, {w: qw, b: qb})

    def on_predict(self, X):
        print(ed.evaluate('mean_absolute_error', data={self.X_: X, self.y_pred: y_test}))


    def samples(self):
        # TODO: posterior sampling
        pass


