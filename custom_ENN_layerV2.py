import tensorflow as tf

class ENNLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation = None, train_center = True, **kwargs):
        super(ENNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.train_center = train_center

    def build(self, input_shape):
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zero',
                                 trainable=True,
                                 name='bias')
        self.w = self.add_weight(shape=(self.units, input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='weight',
                                 constraint=tf.keras.constraints.NonNeg())
        self.centers = self.add_weight(shape=(self.units, input_shape[-1]),
                                       initializer='glorot_uniform',
                                       trainable=self.train_center,
                                       name='center')
        super(ENNLayer, self).build(input_shape)

    def call(self, inputs):
        # Broadcasting to subtract centers from inputs
        squared_X = tf.math.square(tf.expand_dims(inputs, axis=1) - self.centers)  # shape: (batch_size, units, input_dim)
        # Element-wise multiplication and bias subtraction
        weighted_sum = tf.math.reduce_sum(tf.math.multiply(squared_X,self.w), axis=-1)  # shape: (batch_size, units)
        y = weighted_sum - self.b  # shape: (batch_size, units)

        # Apply activation function if provided
        if self.activation is not None:
            y = self.activation(y)

        return y
