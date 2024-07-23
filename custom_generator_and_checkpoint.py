import tensorflow as tf
import pandas as pd
import numpy as np

class DataFrameGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=100, shuffle=False, **kwargs):
        super().__init__(**kwargs)
        """
        Initialization
        :param dataframe: Pandas DataFrame containing the data
        :param x_col: Column name or list of column names to be used as input features
        :param y_col: Column name or list of column names to be used as labels
        :param batch_size: Size of each batch
        :param shuffle: Whether to shuffle the data at the end of each epoch
        """
        self.X= X.values
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()
    
    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: Number of batches per epoch
        """
        return int(np.floor(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: Index of the batch
        :return: Tuple (X, y) containing one batch of data
        """
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X = self.X[batch_indices]
        y = self.y.iloc[batch_indices].values
        
        # Add any preprocessing here if necessary
        
        return X, y
    
    def on_epoch_end(self):
        """
        Updates indices after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indices)



