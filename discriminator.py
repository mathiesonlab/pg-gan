"""
CNN-based discriminator models for pg-gan.
Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang
Date: 2/4/21
"""

# python imports
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, \
    MaxPooling2D, AveragePooling1D, Dropout, Concatenate
from tensorflow.keras import Model

class OnePopModel(Model):
    """Single population model - based on defiNETti software."""

    def __init__(self, pop):
        super(OnePopModel, self).__init__()

        # it is (1,5) for permutation invariance (shape is n X SNPs)
        self.conv1 = Conv2D(32, (1, 5), activation='relu')
        self.conv2 = Conv2D(64, (1, 5), activation='relu')
        self.pool = MaxPooling2D(pool_size = (1,2), strides = (1,2))

        self.flatten = Flatten()
        self.dropout = Dropout(rate=0.5)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.dense3 = Dense(1)#2, activation='softmax') # two classes

        self.pop = pop

    def call(self, x, training=None):
        """x is the genotype matrix, dist is the SNP distances"""
        assert x.shape[1] == self.pop
        x = self.conv1(x)
        x = self.pool(x) # pool
        x = self.conv2(x)
        x = self.pool(x) # pool

        # note axis is 1 b/c first axis is batch
        # can try max or sum as the permutation-invariant function
        #x = tf.math.reduce_max(x, axis=1)
        x = tf.math.reduce_sum(x, axis=1)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return self.dense3(x)

    def build_graph(self, gt_shape):
        """This is for testing, based on TF tutorials"""
        gt_shape_nobatch = gt_shape[1:]
        self.build(gt_shape) # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=gt_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)

class TwoPopModel(Model):
    """Two population model"""

    # integers for num pop1, pop2
    def __init__(self, pop1, pop2):
        super(TwoPopModel, self).__init__()

        # it is (1,5) for permutation invariance (shape is n X SNPs)
        self.conv1 = Conv2D(32, (1, 5), activation='relu')
        self.conv2 = Conv2D(64, (1, 5), activation='relu')
        self.pool = MaxPooling2D(pool_size = (1,2), strides = (1,2))

        self.flatten = Flatten()
        self.merge = Concatenate()
        self.dropout = Dropout(rate=0.5)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.dense3 = Dense(1) # 2, activation='softmax') # two classes

        self.pop1 = pop1
        self.pop2 = pop2

    def call(self, x, training=None):
        """x is the genotype matrix, dist is the SNP distances"""
        assert x.shape[1] == self.pop1 + self.pop2

        # first divide into populations
        x_pop1 = x[:, :self.pop1, :, :]
        x_pop2 = x[:, self.pop1:, :, :]

        # two conv layers for each part
        x_pop1 = self.conv1(x_pop1)
        x_pop2 = self.conv1(x_pop2)
        x_pop1 = self.pool(x_pop1) # pool
        x_pop2 = self.pool(x_pop2) # pool

        x_pop1 = self.conv2(x_pop1)
        x_pop2 = self.conv2(x_pop2)
        x_pop1 = self.pool(x_pop1) # pool
        x_pop2 = self.pool(x_pop2) # pool

        # 1 is the dimension of the individuals
        # can try max or sum as the permutation-invariant function
        #x_pop1_max = tf.math.reduce_max(x_pop1, axis=1)
        #x_pop2_max = tf.math.reduce_max(x_pop2, axis=1)
        x_pop1_sum = tf.math.reduce_sum(x_pop1, axis=1)
        x_pop2_sum = tf.math.reduce_sum(x_pop2, axis=1)

        # flatten all
        #x_pop1_max = self.flatten(x_pop1_max)
        #x_pop2_max = self.flatten(x_pop2_max)
        x_pop1_sum = self.flatten(x_pop1_sum)
        x_pop2_sum = self.flatten(x_pop2_sum)

        # concatenate
        m = self.merge([x_pop1_sum, x_pop2_sum]) # [x_pop1_max, x_pop2_max]
        m = self.fc1(m)
        m = self.dropout(m, training=training)
        m = self.fc2(m)
        m = self.dropout(m, training=training)
        return self.dense3(m)

    def build_graph(self, gt_shape):
        """This is for testing, based on TF tutorials"""
        gt_shape_nobatch = gt_shape[1:]
        self.build(gt_shape) # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=gt_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)

class ThreePopModel(Model):
    """Three population model"""

    # integers for num pop1, pop2, pop3
    def __init__(self, pop1, pop2, pop3):
        super(ThreePopModel, self).__init__()

        # it is (1,5) for permutation invariance (shape is n X SNPs)
        self.conv1 = Conv2D(32, (1, 5), activation='relu')
        self.conv2 = Conv2D(64, (1, 5), activation='relu')
        self.pool = MaxPooling2D(pool_size = (1,2), strides = (1,2))

        self.flatten = Flatten()
        self.merge = Concatenate()
        self.dropout = Dropout(rate=0.5)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.dense3 = Dense(1)#2, activation='softmax') # two classes

        self.pop1 = pop1
        self.pop2 = pop2
        self.pop3 = pop3

    def call(self, x, training=None):
        """x is the genotype matrix, dist is the SNP distances"""
        assert x.shape[1] == self.pop1 + self.pop2 + self.pop3

        # first divide into populations
        x_pop1 = x[:, :self.pop1, :, :]
        x_pop2 = x[:, self.pop1:self.pop1+self.pop2, :, :]
        x_pop3 = x[:, self.pop1+self.pop2:, :, :]

        # two conv layers for each part
        x_pop1 = self.conv1(x_pop1)
        x_pop2 = self.conv1(x_pop2)
        x_pop3 = self.conv1(x_pop3)
        x_pop1 = self.pool(x_pop1) # pool
        x_pop2 = self.pool(x_pop2) # pool
        x_pop3 = self.pool(x_pop3) # pool

        x_pop1 = self.conv2(x_pop1)
        x_pop2 = self.conv2(x_pop2)
        x_pop3 = self.conv2(x_pop3)
        x_pop1 = self.pool(x_pop1) # pool
        x_pop2 = self.pool(x_pop2) # pool
        x_pop3 = self.pool(x_pop3) # pool

        # 1 is the dimension of the individuals
        x_pop1 = tf.math.reduce_sum(x_pop1, axis=1)
        x_pop2 = tf.math.reduce_sum(x_pop2, axis=1)
        x_pop3 = tf.math.reduce_sum(x_pop3, axis=1)

        # flatten all
        x_pop1 = self.flatten(x_pop1)
        x_pop2 = self.flatten(x_pop2)
        x_pop3 = self.flatten(x_pop3)

        # concatenate
        m = self.merge([x_pop1, x_pop2, x_pop3])
        m = self.fc1(m)
        m = self.dropout(m, training=training)
        m = self.fc2(m)
        m = self.dropout(m, training=training)
        return self.dense3(m)

    def build_graph(self, gt_shape):
        """This is for testing, based on TF tutorials"""
        gt_shape_nobatch = gt_shape[1:]
        self.build(gt_shape) # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=gt_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)
