# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:11:28 2020
"""
import MMSGP
from sklearn.base import RegressorMixin, BaseEstimator ,clone
import sklearn.datasets
from sklearn.gaussian_process.kernels import *
from sklearn.preprocessing import normalize,scale
import scipy as sp;
#import matplotlib.pyplot as plt;
#from skbayes.rvm_ard_models import RegressionARD,ClassificationARD,RVR,RVC
from sklearn.svm import SVC
from sklearn.utils import check_X_y;
from sklearn.utils.multiclass import check_classification_targets
from sklearn.multiclass import OneVsRestClassifier as ovr;
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import time 
from scipy import optimize
from sklearn.datasets import make_classification#generate 2d data
from sklearn.manifold import TSNE
import os
import numpy as np
import cvxopt
from cvxopt import matrix, solvers;
from sklearn.metrics.pairwise import pairwise_kernels
from numpy import *
from scipy import *
from sklearn import preprocessing as proc
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
import matplotlib.pyplot as plt
import util
#import fileIO
import tools
#result lists for test,execution time
ourRes=[]
exeTime=[]


eta=80
decayThres=80
decayRate=5

#run 500 AL iterations
for i in range(500):
    #if clause to determine whether activate sparese prior
    if (i>decayThres):
        tem=(decayedEta-decayRate*(eta)/(500-decayThres))
        decayedEta=0 if tem<0 else tem
    else:
        decayedEta=eta
    print('decayedEta: %f' %(decayedEta))
    #create model
    estimator=GPKMC(rbf_scale=1,global_iter=3,gamma_iter=4,C=0.1)
    #create one vs rest wrapper
    mc_model=ovr(estimator,n_jobs=-1)
    t1=time.time()
    #train the model
    mc_model.fit(X[train_index,:],Y[train_index])
    t2=time.time()
    #evaluate test performacne
    score=mc_model.score(X[test_index],Y[test_index])
    ourRes.append(score)
    exeTime.append(t2-t1)
    print('Iter: %d. Score: %f. time: %f' %(i,score,t2-t1))
    #get sample
    sampleIndex , total_mean_var, total_mean_en , max_var= ALSample(X,Y,mc_model,train_index,candidate_index,eta=decayedEta,top=1.5,noise=100)
    print('maxvar: %d'% (max_var))
    #add sample to train set. Note we manipulate the index of data rather than the data themselves.
    train_index=train_index+[sampleIndex]
    #remove sample from candidate pool
    candidate_index=[x for x in candidate_index if x not in [sampleIndex]]