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
class sausage_neurons:
  def __init__(self,q1 = [] , q2 = [], r = []):
    self.q1 = np.vstack(q1)
    self.q2 = np.vstack(q2)
    self.r = np.hstack(r)


####Intra-class clustering algorithm####
from sklearn.cluster import KMeans
def intraclass_cluster(X_i, n_components=10,random_state = None):
    cluster = KMeans(n_clusters = n_components,random_state = random_state)
    X_cluster = cluster.fit_predict(X_i)
    return X_cluster



def cluster_gen(X_i,n_components = 10,random_state = None):
    if len(X_i) > 1:
        cache = intraclass_cluster(X_i,n_components = n_components, random_state = random_state)
        return cache
                
    elif len(X_i) == 1:
        cache = np.array([-1])                         
        return cache
    
    elif len(X_i) == 0 :
        print('empty dataframe were put into the clustering step.')
    

def cluster_generator(X,n_components = 10,random_state = None,verbose = True):
    if verbose:
        start = time.time()
    X_cluster = Parallel(n_jobs=-1)(delayed(cluster_gen)(X_i,n_components = n_components,random_state = random_state) for X_i in X)
    if verbose:
        end = time.time()
        print('Cluster Creation finished at:', end - start)
    return X_cluster


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

####q1 q2 Calculation####
# stretching weight for an intraclass dataset
def q1_q2_calculation(X_i,X_cluster_i):
    X_i = check_data_type_X_npar(X_i)
    q1i = []
    q2i = []
    r = []
    for j in np.unique(X_cluster_i):
        X_ij = X_i[X_cluster_i == j]
        if np.unique(X_ij,axis = 0).shape[0] > 1:
        # check if X_i contains more than 1 unique points
            distances = pdist(X_ij)
            # Convert the condensed distance matrix to a redundant distance matrix
            distance_matrix = squareform(distances)
            max_index = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
            q1i.append(X_ij[max_index[0]])
            q2i.append(X_ij[max_index[1]])
            r.append(np.var(distances))
        else:
        # if X_i contains one point, the default q1, q2 will remains near the point
            q1i.append(X_ij[0] - X_ij[0]/10)
            q2i.append(X_ij[0] + X_ij[0]/10)
            r.append(np.linalg.norm(X_ij[0]/5))
    return q1i, q2i, r

# stretching weight for the whoke dataset
def q_generator(X,X_cluster, verbose = True):
    if verbose:
        start = time.time()
    set_of_qs = Parallel(n_jobs=-1)(delayed(q1_q2_calculation)(X[i],X_cluster[i]) for i in range(len(X)))
    if verbose:
        end = time.time()
        print('q generation finised at:', end-start)
    set_of_q1 = [t[0] for t in set_of_qs]
    set_of_q2 = [t[1] for t in set_of_qs]
    set_of_r = [t[2] for t in set_of_qs]
    return set_of_q1, set_of_q2, set_of_r 

       
####get q1 q2####
def get_q1_q2(X, y,n_components=10,random_state = None, verbose = True):
    # y should already be encoded
    X = data_divider(X, y)
    
    X_copy = deepcopy(X)
    X_cluster = cluster_generator(X_copy, n_components=n_components, random_state = random_state, verbose = verbose)
    set_of_q1, set_of_q2, set_of_r  = q_generator(X,X_cluster, verbose = True)
    
    layer_1 = sausage_neurons(q1=  set_of_q1, q2 = set_of_q2, r = set_of_r)
    return layer_1



####The layer####
## Two layer version
class Sausage_weights:
    def __init__(self,n_components=30,random_state = None, verbose = True):
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(self,X,y):
        X = check_data_type_X_df(X)
        y = check_data_type_y_df(y)

        y_encoded, levels = label_encoder(y)
        q = get_q1_q2(X, y_encoded, n_components=self.n_components,random_state = self.random_state,   
                                        verbose = self.verbose)
        
        self.levels = levels
        self.layer_1 = q
        self.y_encoded = y