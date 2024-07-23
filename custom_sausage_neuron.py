import tensorflow as tf

class SAUSAGE(tf.keras.layers.Layer):
    def __init__(self, units, activation = None, **kwargs):
        super(SAUSAGE, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        

    def build(self, input_shape):
        self.q1 = self.add_weight(shape=(self.units, input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name = 'q1', dtype=tf.float64)
        self.q2 = self.add_weight(shape=(self.units, input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name = 'q2', dtype=tf.float64)
        self.r = self.add_weight(shape=(self.units, ),
                                 initializer='ones',
                                 trainable=True,
                                 name = 'radius',
                                 constraint=tf.keras.constraints.NonNeg(), dtype=tf.float64)
        self.n = input_shape[-1]
        super(SAUSAGE, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float64)
        # the sausage neuron is vectorized in the same fasion as the ENN for a fair comparison, the unmatched performance is probably due to the choice of library
        q12 = tf.math.subtract(self.q2, self.q1)
        # get projection vector k
        k = tf.math.reduce_sum((tf.expand_dims(inputs, axis=1) - self.q1)*tf.expand_dims(q12, axis = 0), axis = 2)
        k = tf.expand_dims(k/tf.expand_dims(tf.norm(q12, axis = 1), axis = 0), axis = -1) * q12 # shape = (batch_size, feature_num)

        # get lambda factor
        temp1 = tf.reduce_sum(k * q12, axis = -1)
        l = tf.zeros_like(temp1, dtype=tf.float64) # automatically the third condition which gives zero
        # now update each element according to the first and second conditions.
        # check first condition
        mask1 = tf.where(temp1 < 0)
        update1 = tf.ones(tf.shape(mask1)[0], dtype=tf.float64)
        l = tf.tensor_scatter_nd_update(l, mask1,update1) # updated lambda
        # check second condition
        k_norm = tf.norm(k, axis = -1)
        temp2 = k_norm / tf.norm(q12, axis = 1)
        mask2 = tf.where(tf.logical_and(temp1 >= 0, temp2 < 1))
        update2 = tf.gather_nd(k_norm, mask2)
        l = tf.tensor_scatter_nd_update(l, mask2, update2) # final lambda tensor with columns represent results from neurons and rows represent each input size = (batch_size, units)
        # calculate y
        y = tf.expand_dims(inputs, axis = 1) - (tf.expand_dims(l, axis = -1) * self.q1 + tf.expand_dims(1-l, axis = -1) * self.q2)
        y = -tf.math.square(tf.norm(y, axis = -1)) # dividend shape = (batch size, units)
        y = y / (self.n * tf.math.square(self.r)) # dividend / nr^2 shape = (batch size, units)

        # Apply activation function if provided
        if self.activation is not None:
            y = self.activation(y)
        else:
            # use gaussian kernel
            y = tf.math.exp(-tf.math.square(x))

        return y
