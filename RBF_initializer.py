# import required library
from copy import deepcopy
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import time
import sys
import warnings
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings("ignore")

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

####Define neurons layer class####
class RBF_neurons:
  def __init__(self,centers = []):
    self.centers = np.vstack(centers)



####Intra-class clustering algorithm####
from sklearn.cluster import KMeans
def intraclass_cluster(X_i, n_components=10,random_state = None):
    cluster = KMeans(n_clusters = n_components,random_state = random_state)
    cluster.fit(X_i)
    return cluster.cluster_centers_



def cluster_gen(X_i,n_components = 10,random_state = None):
    if len(X_i) > 1:
        cache = intraclass_cluster(X_i,n_components = n_components, random_state = random_state)
        return cache
                
    elif len(X_i) == 1:
        cache = np.array([X_i])                         
        return cache
    
    elif len(X_i) == 0 :
        print('empty dataframe were put into the clustering step.')
    

def cluster_generator(X,n_components = 10,random_state = None,verbose = True):
    if verbose:
        start = time.time()
    centers = Parallel(n_jobs=-1)(delayed(cluster_gen)(X_i,n_components = n_components,random_state = random_state) for X_i in X)
    if verbose:
        end = time.time()
        print('Cluster Creation finished at:', end - start)
    return centers


####Label Encoder####
def label_encoder(y):
    labels, levels = pd.factorize(y)
    y = labels
    return y, levels

####Data divider - separate the data to different label####
def data_divider(X, y):
    X = check_data_type_X_df(X)
    y = check_data_type_y_df(y)
    X.reset_index(inplace=True,drop = True)
    y.reset_index(inplace=True,drop = True)
    X_set = []
    for i in np.sort(np.unique(y)):
        X_set.append(X[y == i])
    return X_set
       
####get centers####
def get_centers(X, y,n_components=10,random_state = None, verbose = True):
    # y should already be encoded
    X = data_divider(X, y)
    
    X_copy = deepcopy(X)
    centers = cluster_generator(X_copy, n_components=n_components, random_state = random_state, verbose = verbose)
    
    layer_1 = RBF_neurons(centers =  centers)
    return layer_1



####The layer####
## Two layer version
class RBF_centers:
    def __init__(self,n_components=30,random_state = None, verbose = True):
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(self,X,y):
        X = check_data_type_X_df(X)
        y = check_data_type_y_df(y)

        y_encoded, levels = label_encoder(y)
        centers = get_centers(X, y_encoded, n_components=self.n_components,random_state = self.random_state,   
                                        verbose = self.verbose)
        
        self.levels = levels
        self.layer_1 = centers
        self.y_encoded = y