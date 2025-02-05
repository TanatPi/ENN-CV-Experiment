import tensorflow as tf
tf.random.set_seed(221)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from copy import deepcopy
import pandas as pd
import os
import time
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import json
from RBF_Mahalanobis_no_rotation_initializer import RBF_centers
from custom_RBF_Mahalanobis_neuron_no_rotation import MahalanobisRBFLayer
import warnings
warnings.filterwarnings("ignore")



print(os.getcwd())
#os.chdir('/home/tanat_pi/CNN Protoype/')
os.chdir('W:\DS\Project\CNN Experiment')
#os.chdir('E:\Work\DS\Project\CNN Experiment')

from custom_generator_and_checkpoint import DataFrameGenerator

epochs = 20 # maximum epoch (set at 30 for paper)
num_enn = 10
num_ex = 6 # number of repeated experiments
lr = 0.0002 # learning rate
train_test_splitted = True # if train test is splitted

data = 'CIFAR10'
backbone_model = 'ResNet18'
classification_neuron = 'RBF_Mahalanobis_no_rotation'

#data_directory = 'E:/Work/DS/Project/CNN Experiment/' + backbone_model + '/' + data + '/' # Data directory
data_directory = 'W:/DS/Project/CNN Experiment/' + backbone_model + '/' + data + '/' # Data directory


BATCH_SIZE = 64 # train batch size (64 due to hardware limitation and accuracy is not a goal)     


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


def create_model(n, input_shape, num_class, lr,RBF_initialization):
    initializer = tf.keras.initializers.GlorotUniform()
    #Create model
    input_1 = Input(shape = (input_shape,))
    hidden_1 = MahalanobisRBFLayer(n, R = RBF_initialization.layer_1.R)(input_1)
    output_1 = Dense(num_class, activation='softmax', kernel_initializer=initializer)(hidden_1)
    model = Model(inputs=input_1, outputs=output_1)
    model.compile(optimizer=Adam(learning_rate=lr),
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.TopKCategoricalAccuracy(k=5),tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.F1Score(average = 'micro')])
    model.summary()
    for i in range(len(model.weights)):
        model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)
    RBF_weights = ([RBF_initialization.layer_1.centers,RBF_initialization.layer_1.D])
    model.layers[1].set_weights(RBF_weights)

    return model    

if __name__ == "__main__":
    if train_test_splitted:
        train_df =  pd.read_csv(data_directory + 'extracted_features_train.csv')
        test_df =  pd.read_csv(data_directory + 'extracted_features_test.csv')
        
    else:
        df = pd.read_csv(data_directory + 'extracted_features.csv')
        train_df, test_df = train_test_split(df, test_size=1/6, random_state = 42)

    # Split X and y and get validation set
    train_test_ratio = test_df.shape[0]/train_df.shape[0]
    X_train_val = train_df.drop(columns = 'Class')
    y_train_val = train_df['Class']
    num_class = len(pd.unique(y_train_val))
    input_shape = X_train_val.shape[-1]

    X_test = test_df.drop(columns = 'Class')
    y_test = pd.get_dummies(test_df['Class'])

    # input nodes
    n = num_enn*num_class
    print('number of nodes = ', n)

    
    for i in range(num_ex):
        time_data = []
        accuracy_data = []
        precision_data = []
        recall_data = []
        F1_data = []
        crossentropy_data = []
        top5_accuracy_data = []

        print('training phase', i+1)
        X_train,  X_validation, y_train, y_validation = train_test_split(X_train_val, y_train_val, test_size=train_test_ratio, stratify=y_train_val)

        # initialization
        RBF_initialization = RBF_centers(n_components=num_enn)
        RBF_initialization.fit(X_train,y_train)

        y_train = pd.get_dummies(y_train)
        y_validation = pd.get_dummies(y_validation)
        # generator
        train_generator = DataFrameGenerator(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
        validation_generator = DataFrameGenerator(X_validation, y_validation, batch_size=BATCH_SIZE)
        test_generator = DataFrameGenerator(X_test, y_test, batch_size=BATCH_SIZE)
        
        model = create_model(n, input_shape, num_class, lr,RBF_initialization)


        checkpoint = ModelCheckpoint(data_directory + data + '_' + backbone_model + '_' + classification_neuron + '_' +  f'epochs_{epochs:02d}.keras', verbose=0, monitor='val_categorical_accuracy',save_best_only=True, mode='max')

        start = time.time()
        history = model.fit(
            train_generator,
            epochs=epochs,
            verbose=1,
            validation_data=validation_generator,
            callbacks=[checkpoint])
        end = time.time() 
        time_data.append(end-start)

        print('testing phase', i+1)
        # evaluation
        classification_model = load_model(data_directory + data + '_' + backbone_model + '_' + classification_neuron + '_' +  f'epochs_{epochs:02d}.keras', custom_objects={'RBFLayer': MahalanobisRBFLayer})
        entropy, acc, acc5, pre, rec, f1 = classification_model.evaluate(test_generator)
        accuracy_data.append(acc)
        top5_accuracy_data.append(acc5)
        precision_data.append(pre)
        recall_data.append(rec)
        F1_data.append(f1)
        crossentropy_data.append(entropy)

        tf.keras.backend.clear_session()
        del model, classification_model, train_generator, validation_generator, test_generator
    
        result_dict = {
            "time_used_to_train (s)": time_data,
            "test_accuracy": accuracy_data,
            "test_top5_accuracy": top5_accuracy_data,
            "test_precision": precision_data,
            "test_recall": recall_data,
            "test_F1": F1_data,
            "test_crossentropy": crossentropy_data,
        }   


        

        export_to_json(data_directory + data + '_' + backbone_model + '_' + classification_neuron + f'_maxepochs_{epochs}_learningrate_{lr}_activation_GK_numberofnodes_{n}_' +'results.json.', result_dict)
