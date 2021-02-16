# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:37:50 2019#
@author: None
"""
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
import numpy as np
import os
import fileIO
import tools

def quick_load(data_name): #Quick load some datasets
    if (data_name=='pens'):
        rawX,Y=fileIO.readPenV1(ignoreCase=True)
        X=tools.offset(rawX)
        #Do not include digits£¿-no
        X,Y=tools.removeDigits(X,Y)
        
        X=sp.sparse.load_npz('penV1_downSizeNodigtsX_50.npz')
        ##check imgs
        #tools.plotMatData(X,Y,classType='c',shape=(50,50))
        data=np.load('D:penV1_lowDimXNodigit.npy')
        
        Y=np.array(Y)
        data=normalize(data)
        return data,Y
    if (data_name=='auto'):
        data,Y=fileIO.readAutoDrive(folderPath='D:Sensorless_drive_diagnosis.txt')
        return data,Y
    if (data_name=='yeast'):
        data,Y=fileIO.readYeast(folderPath='D:data.txt')
        return data,Y
    
def feature_selection(X,y):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    clf = ExtraTreesClassifier(n_estimators=20)
    clf.fit(X,y)
    return clf.feature_importances_              
        