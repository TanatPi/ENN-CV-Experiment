import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy



####Data regularization for various function such as transformation, prediction, metrics, etc.####
def check_data_type_X_npar(X):
    # check and convert the input data to default 2D numpy arrays for input
    if type(X) == pd.core.frame.DataFrame:
        # change pandas dataframe to numpy array first
        X = X.values
    elif type(X) == pd.core.series.Series:
        # change pandas series to 2D numpy array first
        X = X.values.reshape(1,len(X))
    elif type(X) == np.ndarray and len(X.shape) == 1:
        # change 1D numpy array to 2D numpy array first
        X = X.reshape(1,len(X))

    return X

def check_data_type_y_npar(y):
    if type(y) == pd.core.series.Series:
        # change pandas series and 1D numpy array to 1D numpy array first
        y = y.values

    return y


def check_data_type_X_df(X):
    # check and convert the input data to default 2D numpy arrays for input
    if type(X) == np.ndarray:
        # change numpy array first to pandas dataframe first
        X = pd.DataFrame(X)
    elif type(X) == pd.core.series.Series:
        # change padas series first to pandas dataframe first
        X = X.to_frame().T.reset_index(drop=True)
    return X

def check_data_type_y_df(y):
    if type(y) == np.ndarray:
        # change pandas series and 1D numpy array to 1D numpy array first
        y = pd.Series(y)
    return y


##input layer##
def input_layer(X,y): 
    labels, levels = pd.factorize(y)
    y = labels
    d,X = getw(X)
    return d,X,y,levels


##First Hidden Layer##
# get weight

def getw(X):
    XX = deepcopy(X)
    d = max(np.diag(np.inner(X,X))) # get d parameter of the data
    radius = np.sqrt(d- np.diag(np.inner(X,X))) # get radius of n-sphere for the data
    XX['radius'] = radius
    return d,XX

def getw_transform(X,d):
    XX = deepcopy(X)
    radius = np.sqrt(d- np.diag(np.inner(X,X))) # get radius of n-sphere for the data
    XX['radius'] = radius
    return XX

# get phi
def getphi(X, y,label):
    d = np.inner(X.iloc[0],X.iloc[0])
    d1 = np.max(np.inner(X[y == label],X[y != label]),axis = 1)
    inner = np.inner(X[y == label],X[y == label])
    mask = inner > np.array([d1]).T
    d2 = np.min(np.where(mask, inner, d),axis = 1)
    
    return np.abs((d1 + d2)/2)

# define neuron class
class neurons:
  def __init__(self,w = [] , p = [], c = 0):
    self.weight = w
    self.bias = p
    self.label = c

def build_first_layer(X,y):
    neuron = pd.DataFrame(X[y == 0].values,getphi(X,y,0))
    stacked_weight = neuron.values
    stacked_bias = neuron.index.values
    label = y[y == 0]
    for i in np.unique(y)[1:]:
        neuron = pd.DataFrame(X[y == i].values,getphi(X,y,i))
        stacked_weight = np.concatenate((stacked_weight,neuron.values),axis = 0)
        stacked_bias = np.concatenate((stacked_bias,neuron.index.values),axis = 0)
        label = np.concatenate((label,y[y == i]),axis = 0)
    layer_1 = neurons(stacked_weight,stacked_bias,label)
    return layer_1

# define transformation with the first layer

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# customized sigmoid
def C_sigmoid(x):
    return 2*sigmoid(x)-1


def activation_function(x):
    output = np.piecewise(x, [x < 0,x >= 0], [0, 1])
    return output

def first_layer_transform_optimization(X,layer):
    return np.sign(np.matmul(X.values,layer.weight.T)-layer.bias) # sign function is used as the activation function 


def first_layer_transform(X,layer):
    return activation_function(np.matmul(X.values,layer.weight.T)-layer.bias) # sign function is used as the activation function 

##First Hidden Layer Optimization##
# Extract indices of the best discreminators:
def layer_indexer(X,y,layer,cutoff = 0):
    
    picked_indices = []
    
    for i in np.unique(layer.label):
        j = -1
        mask = np.ones(len(X[y == i]), dtype=bool)
        n = deepcopy(X[y == i])
        index = []
        layer_1 = neurons()
        layer_1.weight = layer.weight[layer.label == i]
        layer_1.bias = layer.bias[layer.label == i]
        layer_1.label = layer.label[layer.label == i]
        
        original_index = np.arange(len(n))# such that, after masking, we still pick the correct index of the second, third,... best neuron
        
        
        
        while len(n) > 0:
            # sum positive result of the first layer built with label 0 (and 1)
            transformed_n = first_layer_transform_optimization(n,layer_1)
            transformed_n_sign = np.sign(transformed_n-cutoff)
            A = np.sum(transformed_n_sign,axis = 0)
            
            index.append(original_index[A.argsort()[j]]) # keep the index of the best layer
            mask = transformed_n_sign[:,A.argsort()[j]]<0

            # get rid of positively classified points and their neurons
            layer_1.weight = layer_1.weight[mask] # get rid of positively classified points
            layer_1.bias = layer_1.bias[mask]
            layer_1.label = layer_1.label[mask]
            n = n[mask]
            original_index = original_index[mask]
        picked_indices.append(index)
    return picked_indices

# Optimizer
def layer_optimizer(X,y,layer,cutoff = 0):
    layer_opt = neurons()
    layer_index = layer_indexer(X,y,layer,cutoff)

    layer_opt.weight = np.take(layer.weight[layer.label == 0],layer_index[0],axis=0)
    layer_opt.bias = np.take(layer.bias[layer.label == 0],layer_index[0],axis=0)
    layer_opt.label = np.take(layer.label[layer.label == 0],layer_index[0],axis=0)
    
    for i in np.unique(layer.label)[1:]:
        layer_opt.weight = np.append(layer_opt.weight,np.take(layer.weight[layer.label == i],layer_index[i],axis=0),axis=0)
        layer_opt.bias = np.append(layer_opt.bias,np.take(layer.bias[layer.label == i],layer_index[i],axis=0))
        layer_opt.label = np.append(layer_opt.label,np.take(layer.label[layer.label == i],layer_index[i],axis=0))
    return layer_opt

##Output Layer - soft max##
def create_prediction_layer(layer_1):
    weight = []
    for x in np.unique(layer_1.label):
        weight.append((layer_1.label == x)*(1/len(layer_1.label[layer_1.label == x])))
    bias = np.zeros(len(weight))
    layer_2 = neurons(w = np.vstack(weight) , p = bias , c = np.unique(layer_1.label))
    return layer_2

####Second layer transform####
def prediction_layer_transform(X, layer):
    y = np.matmul(X,layer.weight.T)- layer.bias # the activation function is to be decided
    y[np.sum(y,axis = 1) != 0] = y[np.sum(y,axis = 1) != 0]/np.sum(y[np.sum(y,axis = 1) != 0],axis = 1)[:,None]
    y[np.sum(y,axis = 1) == 0] = np.ones(y.shape[1])/y.shape[1]
    return y


###MODEL###
class NSphereNN:
    def __init__(self):
        pass
    
    def fit(self,X,y):
        X = check_data_type_X_df(X)
        y = check_data_type_y_df(y)

        d, X_transformed,y_encoded, levels = input_layer(X,y)
        layer_1 = build_first_layer(X_transformed,y_encoded)
        layer_1 = layer_optimizer(X_transformed,y_encoded,layer_1)
        layer_2 = create_prediction_layer(layer_1)
        
        self.levels = levels
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.d = d
        self.X = X_transformed
        self.y = y_encoded


    def predict(self,X):
        X = check_data_type_X_df(X)
        X = getw_transform(X,self.d)
        X_1 = first_layer_transform(X, self.layer_1) # intermediate output
        prediction = np.argmax(prediction_layer_transform(X_1, self.layer_2 ),axis = 1)
        return self.levels[prediction]
    
    def probability(self,X, print_column = True):
        X = check_data_type_X_df(X)
        X = getw_transform(X,self.d)
        X_1 = first_layer_transform(X, self.layer_1) # intermediate output
        probability = prediction_layer_transform(X_1, self.layer_2)
        if print_column:
            print('columns:', self.levels)
        return probability
    
    def predict_probability(self,X):
        X = check_data_type_X_df(X)
        X = getw_transform(X,self.d)
        X_1 = first_layer_transform(X, self.layer_1) # intermediate output
        probability = prediction_layer_transform(X_1, self.layer_2)
        if probability.shape[0] == 1:
            for i in range(0,probability.shape[1]):
                print(self.levels[i],':',probability[0,i])
        else:
            print('Multiple output is not supported')