import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import ModelCheckpoint
from custom_ENN_layerV2 import ENNLayer
from Ellipsoidal_layer_V5 import EllipsoidLayer
import os
import pandas as pd
import numpy as np
import json


#os.chdir('/home/tanat_pi/CNN Protoype/')
#os.chdir('W:\DS\Project\CNN Protoype')
os.chdir('E:\Work\DS\Project\CNN Experiment')


epochs = 200 # maximum epoch (set at 20 for paper)
num_enn = 10
n = num_enn*10
num_ex = 6 # number of repeated experiments
lr = 0.0002 # learning rate
activation = 'tanh' # activation value
BATCH_SIZE = 64

data_directory = 'W:/DS/Project/CNN Experiment/Shallow Models/'
data = 'CIFAR10'
classification_neuron = 'ENN_no_initialization'

tf.keras.backend.clear_session()

def export_to_json(file_path, new_data):
# for exporting experiment data
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
        
        if isinstance(existing_data, dict) and isinstance(new_data, dict):

            for key, value in new_data.items():
            # Check if the key exists in the original dictionary
                if key in existing_data:
                    # Extend the existing list with the new list
                    existing_data[key].extend(value)
                else:
                    # If the key does not exist, add the new key-value pair
                    existing_data[key] = value
        else:
            raise ValueError("Incompatible data types: existing data and new data must both be dictionaries or both be lists.")

        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
    else:
        with open(file_path, 'w') as file:
            json.dump(new_data, file, indent=4)

if __name__ == "__main__":
    for i in range(num_ex):
        time_data = []
        accuracy_data = []
        precision_data = []
        recall_data = []
        F1_data = []
        crossentropy_data = []
        print('training phase', i+1)
        # Load CIFAR10 dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Normalize the images to a range of 0-1
        X_train = x_train.astype('float32') / 255.0
        X_test = x_test.astype('float32') / 255.0

        # Further split for the validation set
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

        # Create an ImageDataGenerator instance for data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_validation = tf.keras.utils.to_categorical(y_validation, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        # Create generators for the training data
        train_generator = train_datagen.flow(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        validation_generator = train_datagen.flow(
            X_validation,
            y_validation,
            batch_size=BATCH_SIZE
        )

        test_generator = train_datagen.flow(
            X_test,
            y_test,
            batch_size=BATCH_SIZE
        )

        # Build the model
        initializer = tf.keras.initializers.GlorotUniform()
        inputs = Input(shape=(32, 32, 3))
        conv1 = Conv2D(32, (3, 3), activation='relu')(inputs)
        pool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
        pool2 = MaxPooling2D((2, 2))(conv2)
        conv3 = Conv2D(64, (3, 3), activation='relu')(pool2)
        pool3 = MaxPooling2D((2, 2))(conv3)
        flatten_output = Flatten()(pool3)
        # Add classification layers
        hidden_1 = ENNLayer(n, activation=activation)(flatten_output)
        output_1 = tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer)(hidden_1)

        model = Model(inputs=inputs, outputs=output_1)
        model.compile(optimizer=Adam(learning_rate=lr), loss = tf.keras.losses.CategoricalCrossentropy(),
                    metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.F1Score(average = 'micro')])
        model.summary()

        # Training the model with the train generator
        checkpoint = ModelCheckpoint('test.keras', verbose=0, monitor='val_categorical_accuracy',save_best_only=True, mode='max')

                
        start = time.time()
        history = model.fit(
            train_generator,
            epochs=epochs,
            verbose=1,
            validation_data=validation_generator,
            callbacks=[checkpoint])
        end = time.time() 
        time_data.append(end-start)
        # Evaluation
        classification_model = load_model('test.keras', custom_objects={'ENNLayer': ENNLayer})
        entropy, acc, pre, rec, f1 = classification_model.evaluate(test_generator)
        accuracy_data.append(acc)
        precision_data.append(pre)
        recall_data.append(rec)
        F1_data.append(f1)
        crossentropy_data.append(entropy)

        del model, classification_model, train_generator, validation_generator, test_generator

        result_dict = {
            "time_used_to_train (s)": time_data,
            "test_accuracy": accuracy_data,
            "test_precision": precision_data,
            "test_recall": recall_data,
            "test_F1": F1_data,
            "test_crossentropy": crossentropy_data,
        }

        export_to_json(data_directory + 'Shallow_model_' + data + '_' + '_' + classification_neuron + f'_maxepochs_{epochs}_learningrate_{lr}_activation_' + activation + f'_numberofnodes_{n}_' +'results.json.', result_dict)