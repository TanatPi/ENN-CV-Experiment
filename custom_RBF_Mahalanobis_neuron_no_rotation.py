import tensorflow as tf

class MahalanobisRBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, R = [], **kwargs):
        super(MahalanobisRBFLayer, self).__init__(**kwargs)
        self.units = units
        self.rotation = R

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        # Centers for RBF
        self.centers = self.add_weight(name='centers',
                                       shape=(self.units, feature_dim),
                                       initializer='glorot_uniform',
                                       trainable=True)

        # Diagonal matrix (eigenvalues)
        self.diagonal = self.add_weight(name='diagonal',
                                        shape=(self.units, feature_dim),
                                        initializer='ones',  # Diagonal entries start as 1
                                        trainable=True)

    def call(self, inputs):
        expanded_inputs = tf.expand_dims(inputs, axis=1)  # Shape: (batch, units, features)
        diff = expanded_inputs - self.centers  # Shape: (batch, units, features)

        # Compute precision matrix as V * Lambda^-1 * V^T
        inverse_diag = tf.linalg.diag(1.0 / self.diagonal)  # Inverse of the diagonal matrix
        precision_matrices = tf.einsum('uij,ujk->uik', self.rotation, tf.einsum('ukl,ulm->ukm', inverse_diag, self.rotation))

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