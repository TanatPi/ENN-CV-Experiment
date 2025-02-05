import tensorflow as tf

class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.units, input_shape[-1]),
                                       initializer='uniform',
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.units, input_shape[-1]),
                                     initializer='uniform',
                                     trainable=True,
                                     constraint=tf.keras.constraints.NonNeg())
    def call(self, inputs):
        # Compute the RBF activations
        expanded_inputs = tf.expand_dims(inputs, axis=1)
        diff = expanded_inputs - self.centers
        l2 = tf.math.reduce_sum(tf.math.multiply(self.betas,tf.math.square(diff)), axis=-1)
        rbf_outputs = tf.math.exp(-l2)
        return rbf_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(RBFLayer, self).get_config()
        config.update({
            'units': self.units,
        })
        return config