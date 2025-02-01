import tensorflow as tf

class MahalanobisRBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(MahalanobisRBFLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        # Centers for RBF
        self.centers = self.add_weight(name='centers',
                                       shape=(self.units, feature_dim),
                                       initializer='glorot_uniform',
                                       trainable=True)

        # Lower triangular matrix L for Cholesky decomposition
        identity_init = tf.eye(feature_dim, batch_shape=[self.units])  # (units, D, D)
        self.L = self.add_weight(name='cholesky_factors',
                                 shape=(self.units, feature_dim, feature_dim),
                                 initializer=tf.keras.initializers.Constant(identity_init),
                                 trainable=True)

    def call(self, inputs):
        expanded_inputs = tf.expand_dims(inputs, axis=1)  # Shape: (batch, units, features)
        diff = expanded_inputs - self.centers  # Shape: (batch, units, features)

        # Compute precision matrix as L * L^T to ensure positive definiteness
        precision_matrices = tf.einsum('uij,ujk->uik', self.L, tf.linalg.matrix_transpose(self.L))  # (units, D, D)

        # Compute Mahalanobis distance: (x - μ)^T Σ⁻¹ (x - μ)
        mahalanobis_distances = tf.einsum('bui,uij,buj->bu', diff, precision_matrices, diff)  # (batch, units)

        # Compute RBF output
        rbf_outputs = tf.math.exp(-mahalanobis_distances)
        return rbf_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(MahalanobisRBFLayer, self).get_config()
        config.update({'units': self.units})
        return config