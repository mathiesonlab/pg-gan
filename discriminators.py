"""
CNN-based discriminator models for pg-gan.
Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang
Date: 2/4/21
"""

# python imports
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    MaxPooling2D, Dropout, Concatenate
from tensorflow.keras import Model

class PopModel(Model):
    """ A parent class for our individual population models """
    def __init__(self, *pops):
        super().__init__()

        # only supports one, two, and three pop models
        assert len(pops) >= 1 and len(pops) <= 3

        # it is (1,5) for permutation invariance (shape is n X SNPs)
        self.conv1 = Conv2D(32, (1, 5), activation='relu')
        self.conv2 = Conv2D(64, (1, 5), activation='relu')
        self.pool = MaxPooling2D(pool_size = (1,2), strides = (1,2))

        self.flatten = Flatten()
        if len(pops) >= 2:
            # if we include this in 1pop, we'll get an error during
            # summary() as this won't be built
            self.merge = Concatenate()
        self.dropout = Dropout(rate=0.5)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.dense3 = Dense(1) # 2, activation='softmax') # two classes

        self.pops = pops

    def call(self, x, training=None):
        # first divide into populations
        x_pops = []
        start = 0
        for pop in self.pops[:-1]:
            x_pops.append(x[:, start:start+pop, :, :])
            start += pop
        x_pops.append(x[:, start:, :, :])

        # two conv layers for each part
        #### NOTE:
        # for some reason doing it like this:
        #
        # for x_pop in x_pops:
        #     x_pop = self.conv2(x_pop)
        #
        # will not work. Issue specially occurs with Conv2D (ie conv1/2)
        # I think it has to do with x_pop being copied instead of being
        # passed by reference. i am sad that this isn't C
        x_pops = map(self.conv1, x_pops)
        x_pops = map(self.pool, x_pops)
        x_pops = map(self.conv2, x_pops)
        x_pops = map(self.pool, x_pops)

        # 1 is the dimension of the individuals
        x_pops = map(lambda x: tf.math.reduce_sum(x, axis=1), x_pops)

        # # flatten all
        # NOTE: not entirely sure why, but this throws errors if we do a map
        # x_pops = map(self.flatten, x_pops)
        x_pops = [self.flatten(x_pop) for x_pop in x_pops]

        # concatenate
        m = x_pops[0] if len(x_pops) == 1 else self.merge(x_pops)
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

### XXX: NEEDS TESTING
