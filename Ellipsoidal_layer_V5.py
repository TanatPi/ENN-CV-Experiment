# import required library
from copy import deepcopy
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import time
import sys
import warnings
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
# define neuron class
class neurons:
  def __init__(self,wd = [] , p = [] , c = 0):
    self.weights = wd
    self.biases = p
    self.labels = c

# Ellipsoid neurons layer class needed to be defined separately since the basis transformation information need to be stored in addition to weight, bias, and associated class.
# define ellipsoid neuron class
class ellipsoid_neurons:
  def __init__(self,wd = [] , p = [], wc = [] , c = 0, bases = []):
    self.weights = wd
    self.centers = wc
    self.biases = p
    self.labels = c


####Intra-class clustering algorithm####
from sklearn.cluster import KMeans
def intraclass_cluster(X_i, n_components=10,random_state = None):
    cluster = KMeans(n_clusters = n_components,random_state = random_state)
    X_cluster = cluster.fit_predict(X_i)
    return cluster.cluster_centers_, X_cluster



def cluster_gen(X_i,n_components = 10,random_state = None):
    if len(X_i) > 1:
        centers, cache = intraclass_cluster(X_i,n_components = n_components, random_state = random_state)
        return centers, cache
                
    elif len(X_i) == 1:
        centers, cache = X_i.values, np.array([-1])                         
        return centers, cache
    
    elif len(X_i) == 0 :
        print('empty dataframe were put into the clustering step.')
    

def cluster_generator(X,n_components = 10,random_state = None,verbose = True):
    if verbose:
        start = time.time()
    output = Parallel(n_jobs=-1)(delayed(cluster_gen)(X_i,n_components = n_components,random_state = random_state) for X_i in X)
    if verbose:
        end = time.time()
        print('Cluster Creation finished at:', end - start)
    centers = [t[0] for t in output]
    X_cluster = [t[1] for t in output]
    

    return centers, X_cluster


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


####Gram-Schmidt####
def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T
        

####Basis transformation####
def basis_transform(X_i, center):
     return X_i - center

def basis_transform_bulk(X_i, set_of_bases, set_of_center):
    center = np.array([set_of_center]).swapaxes(0,1)
    return np.matmul(np.array(set_of_bases),np.subtract(np.dstack([X_i]*len(set_of_center)).swapaxes(0,1).T,center).swapaxes(1,2))
    # einsum has more floating point -> more precise, but slower
    #return np.einsum('ijk,imk->ijm',np.array(set_of_bases),np.subtract(np.dstack([X_i]*len(set_of_center)).swapaxes(0,1).T,center))

####Label indexer - define after the basis due to set_of_center dependence####
def cluster_label_indexer(set_of_centers,set_of_label, verbose = True):
    # for indexing each cluster to each class
    if verbose:
        start = time.time()
    label_index = np.array([])
    for i in range(len(set_of_label)):
        label_index = np.concatenate((label_index,set_of_label[i]*np.ones(len(set_of_centers[i]))), axis = 0)

    if verbose:
        end = time.time()
        print('Cluster Label indexing finished at:', end-start)
    return label_index

####Weight Calculation####
# stretching weight for an intraclass dataset
def wd_calculation(X_i,X_cluster_i, num_class, centers, max_weight_ratio = 50):
    wd = np.ones((centers.shape[0],centers.shape[-1]))
    X_i = check_data_type_X_npar(X_i)
    for i,j in enumerate(np.unique(X_cluster_i)):
         X_ij = X_i[X_cluster_i == j]
        if np.unique(X_ij,axis = 0).shape[0] > 1:
        # check if X_ij contains more than 1 unique points
            transform_X_ij = X_ij - centers[i]
            max_dis = np.max(transform_X_ij,axis = 0) - np.min(transform_X_ij,axis = 0) # max difference for each axes
            max_dis[max_dis < np.max(max_dis)/max_weight_ratio] = np.max(max_dis)/max_weight_ratio # divide by Zero prevention
            stretching_vector = 2/(max_dis)
            wd[i,:] = stretching_vector**2
    # create scaler factor to match the mean of wd to those of abs(glorot uniform)
    glorot_limit = np.sqrt(6/ (centers.shape[-1] + num_class*centers.shape[0]))
    abs_glorot_mean = glorot_limit/2
    abs_glorot_std = glorot_limit/np.sqrt(12)
    mean_wd = np.mean(wd)
    std_wd = np.std(wd)

    return abs_glorot_std*(wd-mean_wd)/std_wd + abs_glorot_mean

# stretching weight for the whoke dataset
def wd_generator(X,X_cluster, set_of_centers, max_weight_ratio = 50, verbose = True):
    if verbose:
        start = time.time()

    set_of_wd = Parallel(n_jobs=-1)(delayed(wd_calculation)(X[i],X_cluster[i],len(X), np.array(set_of_centers[i]), max_weight_ratio) for i in range(len(X)))
    if verbose:
        end = time.time()
        print('Weight generation finised at:', end-start)
    return set_of_wd

####Bias calculation####
# for an intraclass dataset
def bias_cal(X_i,X_cluster, i, centers, wd, alpha = 0.):
    #transformed_X_other = basis_transform(X_other, centers[i]) # first dimesion is resulting transformation for each neuron, the shape should match the number of neurons.
    X_ij = X_i[X_cluster == np.unique(X_cluster)[i]]

    transformed_X_i = basis_transform(X_i, centers[i])
    transformed_X_ij = basis_transform(X_ij, centers[i])

    #min_other = np.min(np.matmul((transformed_X_other**2),(wd[i]).T))
    intraclass_ellipse = np.matmul((transformed_X_i**2),(wd[i]).T)
    intracluster_ellipse = np.matmul((transformed_X_ij**2),(wd[i]).T)
        
    #max_intraclass = np.max(intraclass_ellipse[intraclass_ellipse<min_other])
    max_intracluster = np.max(intracluster_ellipse)
    max_intraclass = np.max(intraclass_ellipse)



    if not np.isnan(max_intracluster): 
        if alpha < 0 or alpha >1:
            print('invalid alpha')
            sys.exit(1)
        else:
            return alpha*max_intraclass+(1-alpha)*max_intracluster
    else:
        return max_intracluster

def bias_calculator(X_i,X_cluster, centers, wd, bias_cal_parallel= True, alpha = 0.):
 
    # data transformation
    if bias_cal_parallel:
        phi = Parallel(n_jobs = -1)(delayed(bias_cal)(X_i,X_cluster, i, centers, wd, alpha = alpha) for i in range(len(centers)))
    else:
        phi = []
        for i in range(len(centers)):
            phi.append(bias_cal(X_i,X_cluster, i, centers, wd, alpha = alpha))
        
        
    return np.array(phi)

# for the whole dataset
def bias_gen(X,X_cluster, working_label, i,set_of_centers,set_of_wd,bias_cal_parallel, alpha = 0.):

    X_copy = deepcopy(X)
    X_i = deepcopy(X)[working_label[i]]
    X_cluster_i = deepcopy(X_cluster)[working_label[i]]
    X_copy.pop(working_label[i])

    return bias_calculator(X_i,X_cluster_i, np.array(set_of_centers[i]), set_of_wd[i],bias_cal_parallel=bias_cal_parallel, alpha = alpha)
def bias_generator(X,X_cluster, set_of_centers, set_of_wd,working_label, verbose = True, alpha = 0.):
    if verbose:
        start = time.time()
    
    if 1 > len(set_of_centers)/len(working_label):
        bias_cal_parallel = False
        set_of_bias = Parallel(n_jobs = -1)(delayed(bias_gen)(X,X_cluster, working_label, i,set_of_centers,set_of_wd, bias_cal_parallel=bias_cal_parallel, alpha = alpha) for i in range(len(working_label)))
    else:
        bias_cal_parallel = True
        set_of_bias = []
        for i in range(len(working_label)):
            set_of_bias.append(bias_gen(X,X_cluster, working_label, i,set_of_centers,set_of_wd,bias_cal_parallel=bias_cal_parallel, alpha = alpha))
    if verbose:
        end = time.time()
        print('Bias generation finished at:', end - start)
    return set_of_bias
'''
def relu(x):
    a = np.sinh(x)
    output = np.tanh(x)
    # Fix Nan issue
    output[a == np.inf] = 1
    output[a == -np.inf] = -1
    return output

'''

def relu(x):
    # Note that the ellipsoid return negative values for data inside the surface
    #output = (x) * (x >= 0)
    output = x
    return output

####Ellipsoid Layer transformation####

def ellipsoid_transform(X, e_layer):
    X = check_data_type_X_npar(X)
    y = []
    for i in range(len(e_layer.centers)):
        new_X = basis_transform(X, e_layer.centers[i])
        y.append(relu(np.matmul((new_X**2),(e_layer.weights)[i].T) - e_layer.biases[i])) # the activation function is to be decided
    return np.array(y).T

####Transformation for coverage test####
def activation_function_coverage_test(x):
    output = np.piecewise(x, [x <= 0,x > 0], [1, 0])
    return output

def ellipsoid_transform_coverage_test(X, e_layer):
    X = check_data_type_X_npar(X)
    y = []
    for i in range(len(e_layer.centers)):
        new_X = basis_transform(X, e_layer.centers[i])
        y.append(activation_function_coverage_test(np.matmul((new_X**2),(e_layer.weights)[i].T) - e_layer.biases[i])) # the activation function is to be decided
    return np.array(y).T
       
####Create the first ellipsoid layer####
def create_ellipsoid_layer(X, y,n_components=10,random_state = None, max_weight_ratio = 50, verbose = True, max_iter = 1, alpha = 0.):
    # y should already be encoded
    X = data_divider(X, y)
    X_copy = deepcopy(X)
    working_label = np.sort(np.unique(y))

    # perforn only one iteration
    j = 1
    while True:
        print("j =", j)

        set_of_centers_temp, X_cluster = cluster_generator(X_copy, n_components=n_components, random_state = random_state, verbose = verbose)
        set_of_wd_temp =  wd_generator(X_copy,X_cluster, set_of_centers_temp, max_weight_ratio, verbose = verbose)
        set_of_bias_temp = bias_generator(X_copy,X_cluster, set_of_centers_temp, set_of_wd_temp,working_label, verbose = verbose, alpha = 0.)
        set_of_label_temp = cluster_label_indexer(set_of_centers_temp,working_label, verbose = verbose)

        uncovered_X = []
        working_label_to_keep = []
        # check if data is covered
        for i in range(len(X_copy)):
            layer_1_temp = ellipsoid_neurons(wd =  set_of_wd_temp[i], p = set_of_bias_temp[i], wc =  set_of_centers_temp[i], c = working_label[i])
            uncovered_points = X_copy[i][np.sum(ellipsoid_transform_coverage_test(X_copy[i], layer_1_temp),axis = 1)==0]
            if len(uncovered_points) != 0:
                uncovered_X.append(uncovered_points)
                working_label_to_keep.append(True)
                print('Uncovered points = ',len(uncovered_points))
            else:
                working_label_to_keep.append(False)
                print('All points covered')
        working_label = working_label[working_label_to_keep]
        X_copy = deepcopy(uncovered_X)
        
    # get the ellipsoid data
        if j == 1:
            set_of_centers = np.vstack(set_of_centers_temp)
            set_of_wd = np.vstack(set_of_wd_temp)
            set_of_bias = np.hstack(set_of_bias_temp)
            set_of_label = set_of_label_temp

        else:
            set_of_centers = np.concatenate((set_of_centers,np.vstack(set_of_centers_temp)),axis = 0)
            set_of_wd = np.concatenate((set_of_wd,np.vstack(set_of_wd_temp)),axis = 0)
            set_of_bias = np.concatenate((set_of_bias,np.hstack(set_of_bias_temp)),axis = 0)
            set_of_label = np.concatenate((set_of_label,set_of_label_temp),axis = 0)
        
        j += 1
        if (len(X_copy) == 0) or (j > max_iter):
            break

    layer_1 = ellipsoid_neurons(wd =  set_of_wd, p = set_of_bias, wc =  set_of_centers, c = set_of_label)
    return layer_1



####The layer####
## Two layer version
class EllipsoidLayer:
    def __init__(self,n_components=30,random_state = None, max_weight_ratio = 50, verbose = True, max_iter = 1, alpha = 0.):
        self.n_components = n_components
        self.max_weight_ratio = max_weight_ratio
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter
    
    def fit(self,X,y):
        X = check_data_type_X_df(X)
        y = check_data_type_y_df(y)

        y_encoded, levels = label_encoder(y)
        layer_1 = create_ellipsoid_layer(X, y_encoded, n_components=self.n_components,random_state = self.random_state,   
                                         max_weight_ratio = self.max_weight_ratio, verbose = self.verbose, max_iter = self.max_iter, alpha = self.alpha)
        
        self.levels = levels
        self.layer_1 = layer_1
        self.y_encoded = y


    def predict(self,X):
        X = check_data_type_X_df(X)
        X_1 = ellipsoid_transform(X, self.layer_1) # output
        return X_1
