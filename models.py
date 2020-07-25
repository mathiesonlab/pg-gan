"""
CNN-based discriminator models for pg-gan.
Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang
Date: 7/25/20
"""

# python imports
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, \
    MaxPooling1D, AveragePooling1D, Dropout, Concatenate
from tensorflow.keras import Model
from tensorflow.keras import regularizers

class SinglePopModel(Model):
    """Single population model - based on defiNETti software."""

    def __init__(self):
        super(SinglePopModel, self).__init__()

        # it is (1,5) for permutation invariance (shape is n X SNPs)
        self.conv1 = Conv2D(32, (1, 5), activation='relu')
        self.conv2 = Conv2D(64, (1, 5), activation='relu')

        self.flatten = Flatten()

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.dense3 = Dense(2, activation='softmax') # two classes

    def call(self, x):
        """x is the genotype matrix, dist is the SNP distances"""
        x = self.conv1(x)
        x = self.conv2(x)

        # note axis is 1 b/c first axis is batch
        # can try max or sum as the permutation-invariant function
        #x = tf.math.reduce_max(x, axis=1)
        x = tf.math.reduce_sum(x, axis=1)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.dense3(x)

    def build_graph(self, gt_shape):
        """This is for testing, based on TF tutorials"""
        gt_shape_nobatch = gt_shape[1:]
        self.build(gt_shape) # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=gt_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)

class IMmodel(Model):
    """Three population model"""

    def __init__(self, yri, ceu): # integers for num YRI, CEU, CHB
        super(IMmodel, self).__init__()

        # it is (1,5) for permutation invariance (shape is n X SNPs)
        self.conv1 = Conv2D(32, (1, 5), activation='relu')
        self.conv2 = Conv2D(64, (1, 5), activation='relu')

        self.flatten = Flatten()
        self.merge = Concatenate()

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.dense3 = Dense(2, activation='softmax') # two classes

        self.yri = yri
        self.ceu = ceu

    def call(self, x):
        """x is the genotype matrix, dist is the SNP distances"""

        # first divide into populations
        x_yri = x[:, :self.yri, :, :]
        x_ceu = x[:, self.yri:, :, :]

        # two conv layers for each part
        x_yri = self.conv1(x_yri)
        x_ceu = self.conv1(x_ceu)

        x_yri = self.conv2(x_yri)
        x_ceu = self.conv2(x_ceu)

        # 1 is the dimension of the individuals
        # can try max or sum as the permutation-invariant function
        #x_yri_max = tf.math.reduce_max(x_yri, axis=1)
        #x_ceu_max = tf.math.reduce_max(x_ceu, axis=1)
        x_yri_sum = tf.math.reduce_sum(x_yri, axis=1)
        x_ceu_sum = tf.math.reduce_sum(x_ceu, axis=1)

        # flatten all
        #x_yri_max = self.flatten(x_yri_max)
        #x_ceu_max = self.flatten(x_ceu_max)
        x_yri_sum = self.flatten(x_yri_sum)
        x_ceu_sum = self.flatten(x_ceu_sum)

        # concatenate
        m = self.merge([x_yri_sum, x_ceu_sum]) # [x_yri_max, x_ceu_max]
        m = self.fc1(m)
        m = self.fc2(m)
        return self.dense3(m)

    def build_graph(self, gt_shape):
        """This is for testing, based on TF tutorials"""
        gt_shape_nobatch = gt_shape[1:]
        self.build(gt_shape) # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=gt_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)

class OOAmodel(Model):
    """Three population model"""

    # ['YRI', 'CEU', 'CHB']
    def __init__(self, yri, ceu, chb): # integers for num YRI, CEU, CHB
        super(OOAmodel, self).__init__()

        # it is (1,5) for permutation invariance (shape is n X SNPs)
        self.conv1 = Conv2D(32, (1, 5), activation='relu')
        self.conv2 = Conv2D(64, (1, 5), activation='relu')

        self.flatten = Flatten()
        self.merge = Concatenate()

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.dense3 = Dense(2, activation='softmax') # two classes

        self.yri = yri
        self.ceu = ceu
        self.chb = chb

    def call(self, x):
        """x is the genotype matrix, dist is the SNP distances"""

        # first divide into populations
        x_yri = x[:, :self.yri, :, :]
        x_ceu = x[:, self.yri:self.yri+self.ceu, :, :]
        x_chb = x[:, self.yri+self.ceu:, :, :]

        # two conv layers for each part
        x_yri = self.conv1(x_yri)
        x_ceu = self.conv1(x_ceu)
        x_chb = self.conv1(x_chb)

        x_yri = self.conv2(x_yri)
        x_ceu = self.conv2(x_ceu)
        x_chb = self.conv2(x_chb)

        # 1 is the dimension of the individuals
        x_yri = tf.math.reduce_sum(x_yri, axis=1)
        x_ceu = tf.math.reduce_sum(x_ceu, axis=1)
        x_chb = tf.math.reduce_sum(x_chb, axis=1)

        # flatten all
        x_yri = self.flatten(x_yri)
        x_ceu = self.flatten(x_ceu)
        x_chb = self.flatten(x_chb)

        # concatenate
        m = self.merge([x_yri, x_ceu, x_chb])
        m = self.fc1(m) # TODO need to flatten???
        m = self.fc2(m)
        return self.dense3(m)

    def build_graph(self, gt_shape):
        """This is for testing, based on TF tutorials"""
        gt_shape_nobatch = gt_shape[1:]
        self.build(gt_shape) # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=gt_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)
