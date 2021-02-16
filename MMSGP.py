
# -*- coding: utf-8 -*-
"""
Class for MM-SGP.
"""
from sklearn.base import RegressorMixin, BaseEstimator ,clone
import sklearn.datasets
from sklearn.gaussian_process.kernels import *
from sklearn.preprocessing import normalize,scale
import scipy as sp;
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
from tensorflow import keras;
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import MiniBatchKMeans
import util
#import fileIO
import tools

from sklearn.cluster import KMeans

class SGPKMC(BaseEstimator):
    def sig(self,y):#sigmoid function
        return 1.0/(1+np.power(np.e,-y))
    def lam(self,x):
        return +(0.5)*(1.0/x) * self.sig(x)  - 0.25 * (1.0/x)
    def cho_inv(self,A):
        L=sp.linalg.cholesky(A, lower=True)
        S=sp.linalg.cho_solve((L, True),np.diag(np.ones([A.shape[0]])))
        return S
    def objFun(self,thetas):
        S_inv=self.prior_inv+2*einsum('a,ab,ac->bc',self.Lam,self.K,self.K)
        tem=(thetas+0.5)*self.Y_train
        tem2=einsum('i,ij->j',tem,self.K)
        #S=self.cho_inv(S_inv)
        #m_q=einsum('ij,j->i',S,tem2)
        return 0.5*einsum('i,ij,j-> ',tem2,S_inv,tem2)*(+1) 
    def qp_solver(self):
        S_inv=self.prior_inv+2*np.einsum('a,ab,ac->bc',self.Lam,self.K,self.K)+np.diag(np.ones(self.prior_inv.shape[0]))
        #This einsum is not efficient for some reasons
        #Q=einsum('ab,bc,cd->ad',self.Phi_train,S_inv,self.Phi_train.T)*outer(self.Y_train,self.Y_train)
        S_inv=self.cho_inv(S_inv)
        Q=np.linalg.multi_dot([self.K,S_inv,self.K.T])*np.outer(self.Y_train,self.Y_train)
        #Q=0.5*(Q+Q.T)
        p=cvxopt.matrix(np.einsum('b,ab->a',np.ones([self.K.shape[0]]),Q))
        Q=cvxopt.matrix(Q)
        G=cvxopt.matrix(-np.eye(self.K.shape[0]))
        h=cvxopt.matrix(np.zeros(self.K.shape[0]))
        A=cvxopt.matrix(np.ones([self.K.shape[0]])).T
        b=cvxopt.matrix(1.0*self.C)
        solvers.options['show_progress'] = False
        self.qb_debug=[Q, p, G, h, A, b]
        sol=solvers.qp(Q, p, G, h, A, b)
        return np.array(sol['x'])[:,0]
    def __init__(self,C=1.0,fit_intercept=False,label_code=[-1,1],kernel='rbf',track_mq=False,global_iter=2,gamma_iter=5,useJac=True,useHessian=True,gtol=1e-5,xtol=1e-5,barrier_tol=1e-5,use_qp_solver=True,sparseQP=True,qb_verbose=False,rbf_scale=None,activeSet=None):
        #spares candidate is train+candidate, used to provide activeSet.
        self.C=C
        self.fit_intercept=fit_intercept
        self.label_code=label_code
        self.kernel=kernel  

        self.track_mq=track_mq
        if(track_mq):
            self.history_mq=[]            
        self.global_iter=global_iter
        self.gamma_iter=gamma_iter
        self.useJac=useJac
        self.useHessian=useHessian
        ###the following 3 are not used since sp.optimize.minimize() can not identify these arguments
        self.gtol=gtol
        self.xtol=xtol
        self.barrier_tol=barrier_tol 
        #fit should only include x and y. Put other args to _init.
        self.use_qp_solver=use_qp_solver#use a qp solver or general minmizer solver.
        self.sparseQP=sparseQP
        self.qb_verbose=qb_verbose
        #rbf kernel's parameter
        self.rbf_scale=rbf_scale
        self.activeSet=activeSet

    def get_activeSet(self,X,method='kmean',newX=None,sparseRate=0.2,dataSize=1):
        if (newX is None):
            #Then use k mean over X to get act set
            #set cluster num = X.features*5
            if(method=='kmean'):
                kms=KMeans(n_clusters=int(dataSize*sparseRate)+1)
                kms.fit(X)
                res=kms.cluster_centers_
                self.activeSet=res
            return res
        else:#preserved for incremental training
            return None

    def fit(self,X,y):#binary classifier. ovr classifier dealt by sklearn
        #check labels
        binarizor=LabelBinarizer(neg_label=self.label_code[0], pos_label=self.label_code[1])
        self.binarizor=binarizor;
        t00=time.time()
        if(self.activeSet is None):
            self.activeSet=X
        self.X_train=X
        self.Y_train=y
        if( self.kernel == 'linear'):
            dotkernel=DotProduct(sigma_0=self.rbf_scale)
            self.prior_inv=dotkernel(self.activeSet)
            self.K=dotkernel(self.X_train,self.activeSet)
        else:
            self.prior_inv=pairwise_kernels(self.activeSet,metric='rbf',gamma=self.rbf_scale)
            self.K=pairwise_kernels(self.X_train,self.activeSet,metric='rbf',gamma=self.rbf_scale)
        m=self.X_train.shape[0]
        self.gamma=ones([m])
        self.Lam=array(list(map(self.lam,self.gamma)))
        self.S_inv=self.prior_inv+2*einsum('a,ab,ac->bc',self.Lam,self.K,self.K)
        self.m_q=zeros([m])    
        self.thetas=ones([m])
        constraint=sp.optimize.LinearConstraint(ones([m]),[self.C],[self.C])
        bounds=sp.optimize.Bounds(zeros([m]),zeros([m])+np.inf)
        t0=time.time()
        self.head=t0-t00
        if(self.use_qp_solver is False):
            pass
        for ii in range(self.global_iter):
            #print('GLO iteration:  '+str(ii))
            t1=time.time()
            if(self.use_qp_solver is False):
                res=sp.optimize.minimize(self.objFun,self.thetas,method='trust-constr',constraints=constraint,bounds=bounds,jac='2-point',hess=sp.optimize.BFGS())
                self.thetas=res.x
                self.optRes=res
            else:
                self.thetas=self.qp_solver()                
            t2=time.time()
            self.qptime=t2-t1
            for jj in range(self.gamma_iter):
                t5=time.time()
                self.Lam=array(list(map(self.lam,self.gamma)))
                #print(Lam.shape)
                self.S_inv=self.prior_inv+2*einsum('a,ab,ac->bc',self.Lam,self.K,self.K)+np.diag(np.ones(self.prior_inv.shape[0]))#add diag term to guarantee inverse
                tem=(self.thetas+0.5)*y
                self.m_augpart=tem
                tem2=einsum('i,ij->j',tem,self.K)
                t3=time.time()
                S=self.cho_inv(self.S_inv)
                self.S_q=S
                t4=time.time()
                #print('compute inverse in '+str(t4-t3) + ' secs')
                
                self.m_q=einsum('ij,j->i',S,tem2)
                middleTerm=einsum('a,b->ab',self.m_q,self.m_q)+S#  S + m * m^T
                #update gamma
                self.gamma= sqrt (einsum('ab,bc,ac->a',self.K,middleTerm,self.K) ) #gamma square_i=phi_i^T (middleterm) phi_i
            t6=time.time()
            self.gammatime=t6-t2

    def predict(self,X_test):
        if( self.kernel == 'linear'):
            dotkernel=DotProduct(sigma_0=self.rbf_scale)
            testPhi=dotkernel(X_test,self.activeSet)
        else:
            testPhi=pairwise_kernels(X_test,self.activeSet,metric='rbf',gamma=self.rbf_scale)
        #testPhi=pairwise_kernels(X_test,self.X_train,metric='rbf',gamma=self.rbf_scale)
        prediction=einsum('a,ba->b',self.m_q,testPhi)   
        probability=array(list(map(self.sig,prediction)))
        for i in range(len(probability)):
            if(probability[i]>0.5):
                probability[i]=1
            else:
                probability[i]=0
        return probability
    
    def predict_proba(self,X_test):
        if( self.kernel == 'linear'):
            dotkernel=DotProduct(sigma_0=self.rbf_scale)
            testPhi=dotkernel(X_test,self.activeSet)
        else:
            testPhi=pairwise_kernels(X_test,self.activeSet,metric='rbf',gamma=self.rbf_scale)
        prediction=einsum('a,ba->b',self.m_q,testPhi)   
        probability=array(list(map(self.sig,prediction)))
        res=zeros([X_test.shape[0],2])
        res[:,1]=probability
        res[:,0]=1-probability
        return res
    
    def predict_var(self,X_test,useAug=True,sampleSize=200,anchor=None,testMode=False,noise=1):
        if (useAug is False):
            #need to be changed later, none aug mode is not implemented
            if( self.kernel == 'linear'):
                dotkernel=DotProduct(sigma_0=self.rbf_scale)
                testPhi=dotkernel(X_test,self.X_train)
            else:
                testPhi=pairwise_kernels(X_test,self.X_train,metric='rbf',gamma=self.rbf_scale)

            var=np.einsum('ab,bb,ba->a',testPhi,self.S_q,testPhi.T)
            return var
        else:
            if(anchor is None):
                print('Please provide anchor set for augmented variance prediction')
                return None

            if( self.kernel == 'linear'):
                dotkernel=DotProduct(sigma_0=self.rbf_scale)
                K_TA=dotkernel(self.activeSet,anchor)
                K_AA=dotkernel(anchor,anchor)
            else:
                K_TA=pairwise_kernels(self.activeSet,anchor,metric='rbf',gamma=self.rbf_scale)
                K_AA=pairwise_kernels(anchor,anchor,metric='rbf',gamma=self.rbf_scale)#np.diag(np.ones(anchor.shape[0]))
            K=self.K
            K_AT=K_TA.T
            self.prior=self.cho_inv(self.prior_inv+np.diag(np.ones(self.prior_inv.shape[0]))*0.1)
            kStar=np.einsum('ab,bc,cd->ad',self.K,self.prior,K_TA)
            kStarStar=np.einsum('ab,ac->bc',kStar,kStar)
            UR_block=np.einsum('ab,bc->ac',K.T,kStar)+K_TA
            
            LR_block=np.einsum('ab,bc->ac',K_AT,K_TA)+kStarStar
            #upper half:
            upper=np.concatenate((self.S_inv,UR_block),axis=1)
            #lower half:
            lower=np.concatenate((UR_block.T,LR_block),axis=1)
            #full augmented posterior
            S_aug=np.concatenate((upper,lower),axis=0)
            S_aug+=np.diag(np.ones(S_aug.shape[0]))*noise
            S_aug=np.linalg.inv(S_aug)
            S_aug+=np.diag(np.ones(S_aug.shape[0]))*noise

            if( self.kernel == 'linear'):
                dotkernel=DotProduct(sigma_0=self.rbf_scale)
                testPhi_aug=dotkernel(X_test,np.concatenate((self.activeSet,anchor),axis=0))
                #K_aug=dotkernel(self.activeSet,np.concatenate((self.activeSet,anchor),axis=0))
            else:
                testPhi_aug=pairwise_kernels(X_test,np.concatenate((self.activeSet,anchor),axis=0),metric='rbf',gamma=self.rbf_scale)
                #K_aug=pairwise_kernels(self.activeSet,np.concatenate((self.activeSet,anchor),axis=0),metric='rbf',gamma=self.rbf_scale)
            
            K_aug=np.concatenate((K.T,kStar.T),axis=0)
            m_aug=einsum('i,ji->j',self.m_augpart,K_aug)
            m_aug=einsum('ab,b->a',S_aug,m_aug)
            #sample and put into sigmoid
            #draw 1000 samples and compute the sig val
            sigStac=[]
            sample_value=[]
            for i in range(sampleSize):
                aSamp=sp.stats.multivariate_normal.rvs(mean=m_aug,cov=S_aug)
                #self.sig
                sample_val=einsum('a,ba->b',aSamp,testPhi_aug)
                sample_value.append(sample_val)
                sigStac.append(sig(sample_val))
            sigStac=np.array(sigStac)
            sampleVar=np.var(sigStac,axis=0)
            if testMode:
                return sampleVar,sigStac,np.array(sample_value)
            return sampleVar    
    def score(self,X_test,Y_test):        
        res = self.predict(X_test)
        accuracy=sum(np.logical_not(np.logical_xor(Y_test,res)))/len(res) 
        print(accuracy)
        return accuracy

            
    def accu(self,X_test,Y_test):
        probability = self.predict(X_test)
        tem=probability.copy()
        for i in range(len(tem)):
            if(tem[i]>0.5):
                tem[i]=1
            else:
                tem[i]=0
        accuracy=sum(np.logical_not(np.logical_xor(Y_test,tem)))/len(tem)                
        print(accuracy)
        return accuracy
    
    def getNcluster(self,trainInd):#use the current len to deterimine the level of sparsity
        if(len(trainInd)<100):
            return len(trainInd)
        else:
            return int(len(trainInd)*0.2)#else use 20% of the training
    
def ALSample(X,Y,mc_model,train_index,candidate_index,eta=None,top=1,noise=1,anchorNum=10,sampleSize=50):
    #number of classes
    k=len(np.unique(Y))
    if (eta is None):
        eta=10*k
    #get en
    proba=mc_model.predict_proba(X[candidate_index,:])
    en=[]
    for j in range(proba.shape[0]):
        en.append(sp.stats.entropy(proba[j,:]))
    estimators=mc_model.estimators_
    rbf_scale=estimators[0].rbf_scale
    var=np.zeros(len(candidate_index))
    #adjust the lenscale and top
    if(rbf_scale!=0):
        adjTop=1/pow(10,rbf_scale)
    else:
        adjTop=3
    #get anchor 
    anchorInd=OODIdentify(X[train_index,:],X[candidate_index,:],sampleSize=anchorNum,rbf_scale=rbf_scale,top=adjTop,kernel=estimators[0].kernel)
    #a,p1,p2=OODIdentify(X[train_index,:],X[candidate_index,:],sampleSize=10,rbf_scale=rbf_scale,top=0.5,testMode=True)
    anchor=X[candidate_index,:][anchorInd,:]
    #a,b,c=estimators[0].predict_var(X[candidate_index,:],useAug=True,anchor=anchor,sampleSize=200,testMode=True)
    if(eta!=0):
        for estimator in estimators:
            var=var+estimator.predict_var(X[candidate_index,:],useAug=True,anchor=anchor,sampleSize=sampleSize,noise=noise)
                    
        weightedVar=var*eta/k
        sampleScore=en+weightedVar
    else:
        sampleScore=en
    max_var=np.max(var)
    total_mean_var=np.sum(var)/len(var)
    total_mean_en=np.sum(en)/len(en)
    sampleIndex=np.argmax(sampleScore)#the index of whole training set,X            
    print('max Score: '+str(sampleScore[sampleIndex])+' en:'+ str(en[sampleIndex])+ ' var: '+ str(var[sampleIndex]))
    #max_var=np.max(var)
    return candidate_index[sampleIndex],total_mean_var,total_mean_en,max_var

    

def plot2D(x,y,colors=['k','b','r'],truePdfs=None,predictAlpha=None,model=None,binaryClassifier=True,plotType='truePost',anchor=None,trainInd=[0],testInd=[0],candInd=[0],plotIter=0,plotFreq=1,acc=0,cf=0,oodInd=None,mismatchOODInd=None,saveDataCode=None,saveFig=False,path=None,cmap='coolwarm',title='Fig',close=True,dpi=100,figsize=(25,16)):
    n_grid = 500
    max_x    = np.max(x,axis = 0)
    min_x    = np.min(x,axis = 0)
    XX1         = np.linspace(min_x[0]*1,max_x[0]*1,n_grid)
    XX2         = np.linspace(min_x[1]*1,max_x[1]*1,n_grid)
    x1,x2      = np.meshgrid(XX1,XX2)
    Xgrid      = np.zeros([n_grid**2,2])
    Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
    Xgrid[:,1] = np.reshape(x2,(n_grid**2,))
    Ygrid=np.zeros([Xgrid.shape[0],len(np.unique(y))])
    #get data last ind is the candidate added from last AL iter
    if(plotFreq!=1):
        xTrain=x[trainInd[:-plotFreq],:]
        yTrain=y[trainInd[:-plotFreq]]
    else:
        xTrain=x[trainInd,:]
        yTrain=y[trainInd]    
    xTest=x[testInd,:]
    yTest=y[testInd]
    xSample=x[trainInd[-plotFreq:],:]
    ySample=y[trainInd[-plotFreq:]]   
    xCand=x[candInd,:]
    yCand=y[candInd]
    if(plotType=='dec'):   
        trueEn=model.predict(Xgrid)       
    elif(plotType=='proba'):
        trueEn=model.predict_proba(Xgrid)[:,0]      
    elif(plotType=='predictEn'):
        trueEn=sp.stats.entropy(model.predict_proba(Xgrid).T)
    elif(plotType=='augVar'):        
        trueEn=model.predict_var(Xgrid,useAug=True,anchor=anchor,sampleSize=200)
    elif(plotType=='var'):
        trueEn=model.predict_var(Xgrid,useAug=False,anchor=anchor,sampleSize=200)
    elif(plotType=='gpvar'):
        model.predict_proba(Xgrid)
        trueEn=model.base_estimator_.var
    #plot true class distri(include decision boundary)
    a=plt.figure(dpi=dpi,figsize=figsize)
    plt.contourf(XX1,XX2,np.reshape(trueEn,(n_grid,n_grid)),cmap=cmap)
    cb = plt.colorbar(pad=0.01)
    cb.ax.tick_params(labelsize=40)
    plt.plot(xCand[yCand==0,0],xCand[yCand==0,1],"yo", markersize = 14,label='candidate_positive')
    plt.plot(xCand[yCand==1,0],xCand[yCand==1,1],"rs", markersize = 14,label='candidate_positive')
    plt.plot(xTrain[yTrain==0,0],xTrain[yTrain==0,1],"k^", markersize = 28,label='train_positive')         
    plt.title(title,fontsize=50)
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tick_params(axis='both', which='minor', labelsize=40)
    plt.tight_layout()
    if (saveFig):
        plt.savefig(path+plotType+str(plotIter+1)+'iter.png',bbox_inches='tight',dpi=dpi)
        plt.savefig(path+plotType+str(plotIter+1)+'iter.pdf',bbox_inches='tight',dpi=dpi)
        plt.savefig(path+plotType+str(plotIter+1)+'iter.eps',bbox_inches='tight',dpi=dpi)
        if(close):
            plt.close(a)
    return    
def syntheticDataGen(means=None,covs=None,samplePerClass=500,delta=2,outlierSize=50,outlierDelta=1):#generate 2 classes
    istroVar=.1*delta*delta
    istroVar2=.1*outlierDelta*outlierDelta
    if(means==None):
        a1=np.random.multivariate_normal([5.5,2.5], [[istroVar,0],[0,istroVar]], samplePerClass)
        a2=np.random.multivariate_normal([5.5,1.5], [[istroVar,0],[0,istroVar]], samplePerClass)
        b2=np.random.multivariate_normal([5.5,4.5], [[istroVar2,0],[0,istroVar2]],outlierSize )        
        #b2=np.random.multivariate_normal([5.5,7.5], [[istroVar2,0],[0,istroVar2]],outlierSize )        
        b1=np.random.multivariate_normal([5.5,-1.5], [[istroVar2,0],[0,istroVar2]],outlierSize )        
        #b1=np.random.multivariate_normal([5.5,-4.5], [[istroVar2,0],[0,istroVar2]],outlierSize )        
        #plt.plot(x[:,0], x[:,1], 'x')
        x=np.concatenate([a1,b1,a2,b2],axis=0)
        #Create pdfs
        y=np.zeros((samplePerClass+outlierSize)*2)
        for i in range(2):
            y[i*(samplePerClass+outlierSize):(i+1)*(samplePerClass+outlierSize)]=i
        #ydim=np.argmax(y,axis=1)
        plt.plot(x[y==0,0],x[y==0,1],"ro", markersize = 3)
        plt.plot(x[y==1,0],x[y==1,1],"ks", markersize = 3)
        return x,y
    
#test real dataset
def syntheticDataGen2(means=None,covs=None,samplePerClass=500,delta=2,outlierSize=50,outlierDelta=1):#generate Mix Gaussian synthetic data 
    #like in prior nn, use delta to control overlap
    istroVar=.1*delta*delta
    istroVar2=.1*outlierDelta*outlierDelta
    #class weight in gmm
    largeWeight=samplePerClass/(samplePerClass*3+outlierSize*3)
    smallWeight=outlierSize/(samplePerClass*3+outlierSize*3)
    if(means==None):
        a1=np.random.multivariate_normal([2.5,2.5], [[istroVar,0],[0,istroVar]], samplePerClass)
        a2=np.random.multivariate_normal([.5,0.5], [[istroVar,0],[0,istroVar]], samplePerClass)
        a3=np.random.multivariate_normal([4.5,0.5], [[istroVar,0],[0,istroVar]], samplePerClass)
        b3=np.random.multivariate_normal([-2,8.5], [[istroVar2,0],[0,istroVar2]],outlierSize )        
        b2=np.random.multivariate_normal([8,8.5], [[istroVar2,0],[0,istroVar2]],outlierSize )        
        b1=np.random.multivariate_normal([2.5,-6.5], [[istroVar2,0],[0,istroVar2]],outlierSize )        
        #plt.plot(x[:,0], x[:,1], 'x')
        x=np.concatenate([a1,b1,a2,b2,a3,b3],axis=0)
        #Create pdfs
        n1=multivariate_normal(mean=[2.5,2.5], cov=[[istroVar,0],[0,istroVar]])
        n2=multivariate_normal(mean=[.5,.5], cov=[[istroVar,0],[0,istroVar]])
        n3=multivariate_normal(mean=[4.5,0.5], cov=[[istroVar,0],[0,istroVar]])
        nn3=multivariate_normal([-2,8.5], [[istroVar2,0],[0,istroVar2]])
        nn2=multivariate_normal([8,8.5], [[istroVar2,0],[0,istroVar2]])
        nn1=multivariate_normal([2.5,-6.5], [[istroVar2,0],[0,istroVar2]])
        y=np.zeros([(samplePerClass+outlierSize)*3,3])
        # true class distri of the samples(not normalized)
        trueProb=np.zeros([(samplePerClass+outlierSize)*3,3])#we wont need this
        for i in range(3):
            y[i*(samplePerClass+outlierSize):(i+1)*(samplePerClass+outlierSize),i]=1 
        trueEn=sp.stats.entropy(trueProb.T)#we wont need this
        #plot true post here once for all
        n_grid = 500
        max_x      = np.max(x,axis = 0)
        min_x      = np.min(x,axis = 0)
        XX1         = np.linspace(min_x[0],max_x[0],n_grid)
        XX2         = np.linspace(min_x[1],max_x[1],n_grid)
        x1,x2      = np.meshgrid(XX1,XX2)
        Xgrid      = np.zeros([n_grid**2,2])
        Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
        Xgrid[:,1] = np.reshape(x2,(n_grid**2,))
        trueProb=np.zeros([Xgrid.shape[0],3])
        pdfs=[[n1,nn1],[n2,nn2],[n3,nn3]]
        for i in range(3):
            trueProb[:,i]=pdfs[i][0].pdf(Xgrid)*largeWeight+pdfs[i][1].pdf(Xgrid)*smallWeight
        #get the entropy of each grid            
        trueEn=sp.stats.entropy(trueProb.T)  
        a=plt.figure(figsize=(25,16))
        plt.contourf(XX1,XX2,np.reshape(trueEn,(n_grid,n_grid)),cmap="coolwarm",figsize = (20,12))
        plt.colorbar()
        ydim=np.argmax(y,axis=1)
        plt.plot(x[ydim==0,0],x[ydim==0,1],"ro", markersize = 3)
        plt.plot(x[ydim==1,0],x[ydim==1,1],"ks", markersize = 3)
        plt.plot(x[ydim==2,0],x[ydim==2,1],"b^", markersize = 3)
        return x,y,trueProb,trueEn,pdfs
    else:
        return None    


def OODIdentify(train,candi,sampleSize=5,rbf_scale=1.0,top=0.5,testMode=False,kernel=None):
    if(kernel=='rbf'):
        kernelDist=pairwise_kernels(train,candi,metric='rbf',gamma=rbf_scale)
        kernelDist2=pairwise_kernels(candi,candi,metric='rbf',gamma=rbf_scale)
    else:
        dotkernel=DotProduct(sigma_0=rbf_scale)
        kernelDist=dotkernel(train,candi)
        kernelDist2=dotkernel(candi,candi)
    p1=np.sum(kernelDist,axis=0)/train.shape[0]
    p2=np.sum(kernelDist2,axis=0)/candi.shape[0]
    diff=top*p2-p1
    if(max(diff)<0):
        OODIndex=[argmax(diff)]
    else:
        OODIndex=list(argsort(diff)[::-1][0:sampleSize]) 
    if (testMode):
        return OODIndex,p1,p2
    else:    
        return OODIndex

def sig(y):#sigmoid function
    return 1.0/(1+np.power(np.e,-y))
def lam(x):
    return +(0.5)*(1.0/x) * sig(x)  - 0.25 * (1.0/x)
def cho_inv(A):
    L=sp.linalg.cholesky(A, lower=True)
    S=sp.linalg.cho_solve((L, True),np.diag(np.ones([A.shape[0]])))
    return S


def get_activeSet(X,method='kmean',newX=None,sparseRate=0.2,dataSize=1,previousModel=None):
        if (newX is None):
            if(method=='kmean'):
                if(sparseRate>1):
                    kms=KMeans(n_clusters=sparseRate-1)
                else:
                    kms=KMeans(n_clusters=int(dataSize*sparseRate)-1)
                kms.fit(X)
                res=kms.cluster_centers_
                return res,None
            elif(method=='partialKmean'):
                if(previousModel is None):
                    kms=MiniBatchKMeans(n_clusters = int(dataSize*sparseRate)-1)
                    kms.fit(X)
                    res=kms.cluster_centers_
                    return res, kms
                else:
                    previousModel.partial_fit(X[-1,:].reshape(1,-1))
                    res=previousModel.cluster_centers_
                    return res, previousModel

def computeSparseLevel(decayRate='linear',iterOffset=0,currentIter=0,targetSparseLevel=0.5,targetIter=500):
    if(decayRate == 'linear'):
        return 1-iterOffset*(1-targetSparseLevel)/(targetIter-currentIter)


def GPAL(X,Y,train_ind,candidate_ind,test_ind,sample='En',kernel='rbf',Niter=500,eta=10):
    ourRes=[]
    train_index=train_ind.copy()
    test_index=test_ind.copy()
    candidate_index=candidate_ind.copy()
    varRes=[]
    enRes=[]
    for i in range(Niter):
        print(i)
        if(kernel=='linear'):
            dotkernel=DotProduct(sigma_0=1)
            model=GPC(kernel=dotkernel)
        else:
            model=GPC()
        model.fit(X[train_index],Y[train_index])
        ourRes.append(model.score(X[test_index,:],Y[test_index]))
        print(ourRes[-1])
        if(sample=='rand'):
            sampleIndex=np.random.randint(len(candidate_index))
        elif(sample=='En'):            
            proba=model.predict_proba(X[candidate_index,:])
            en=sp.stats.entropy(proba.T)
            sampleScore=en
            sampleIndex=np.argmax(sampleScore)
        elif(sample=='var'):
            model.predict_proba(X[candidate_index,:])
            meanVar=np.zeros(len(candidate_index))
            for tem in model.base_estimator_.estimators_:
                meanVar=meanVar+tem.var
            sampleIndex=np.argmax(meanVar)
        elif(sample=='varEN'):
            proba=model.predict_proba(X[candidate_index,:])
            en=sp.stats.entropy(proba.T)
            meanVar=np.zeros(len(candidate_index))
            enRes.append(np.mean(en))
            
            for tem in model.base_estimator_.estimators_:
                meanVar=meanVar+tem.var
            sampleIndex=np.argmax(meanVar/len(np.unique(Y))*eta + en)   
            varRes.append(np.mean(meanVar))                 
            print('max var %f----selected var %f-----selected en %f '%(np.max(meanVar),meanVar[sampleIndex],en[sampleIndex]))
        sampleIndex=candidate_index[sampleIndex]
        train_index=train_index+[sampleIndex]
        candidate_index=[x for x in candidate_index if x not in [sampleIndex]]
    return [ourRes,varRes,enRes]


def SGPMED_Rand(X,Y,train_ind,candidate_ind,test_ind,decayThres=100,decayRate=3,eta=80,sparseThres=100,sparseRate=0.5,sparseUpdateFreq=1,rbf_scale=0.2,global_iter=2,gamma_iter=3,C=0.001,kernel='linear',anchorNum=10,top=0.5,noise=1,njob=1,actSet='fromTrain',smoothSparse = 'fixed',sparseUpdate='kmean',checkPoison=False):
    ourRes=[]
    train_index=train_ind.copy()
    test_index=test_ind.copy()
    candidate_index=candidate_ind.copy()
    for i in range(500):
        estimator=SGPKMC(rbf_scale=rbf_scale,global_iter=global_iter,gamma_iter=gamma_iter,C=C,kernel=kernel,activeSet=None)
        mc_model=ovr(estimator,n_jobs=njob)
        t1=time.time()
        mc_model.fit(X[train_index,:],Y[train_index])
        t2=time.time()
        score=mc_model.score(X[test_index],Y[test_index])
        ourRes.append(score)
        print('iter %d;acc:%f' %(i,ourRes[-1]))
        sampleIndex=np.random.randint(len(candidate_index))
        sampleIndex=candidate_index[sampleIndex]
        train_index=train_index+[sampleIndex]
        candidate_index=[x for x in candidate_index if x not in [sampleIndex]]
    return ourRes
    
def SGPMED_AL(X,Y,train_ind,candidate_ind,test_ind,decayThres=100,decayRate=3,eta=80,sparseThres=100,sparseRate=0.5,sparseUpdateFreq=1,rbf_scale=0.2,global_iter=2,gamma_iter=3,C=0.001,kernel='linear',anchorNum=10,top=0.5,noise=1,njob=1,actSet='fromTrain',smoothSparse = 'fixed',sparseUpdate='kmean',checkPoison=False,Niter=500):    
    ourRes=[]
    var_mean=[]
    en=[]
    var_max=[]
    activeSet=None
    iniSize=len(train_ind)
    train_index=train_ind.copy()
    test_index=test_ind.copy()
    candidate_index=candidate_ind.copy()
    poisonSamples=[]#the index of poison sample
    previousKMEANS=None
    iterOffset=0
    i=0
    while (i<Niter):
        if(actSet=='fromTrain'):
            sparseCand=X[train_index,:]
        else:
            sparseCand=X[train_index+candidate_index,:]                
        if (i>decayThres):
            tem=(decayedEta-decayRate*(eta)/(500-decayThres))
            decayedEta=0 if tem<0 else tem
        else:
            decayedEta=eta
        print('decayedEta: %f' %(decayedEta))
        if(len(train_index)>sparseThres and len(train_index)%sparseUpdateFreq==0):
            iterOffset+=1
            if (smoothSparse == 'fixed'):#then the size of sparse gram fixed to sparseThres after training size>sparseThres
                activeSet=get_activeSet(sparseCand,dataSize=len(train_index),sparseRate=sparseRate)[0]
                print('active set has size of:'+str(activeSet.shape))
            elif(smoothSparse == 'linearDecay'):#in this case the sparsity level will gradually drop from 100%to sparseRate in linear rate. We also paritially fit k-means to make it more stable
                #compute the sparseLevel
                adjRate=computeSparseLevel(decayRate='linear',iterOffset=iterOffset,currentIter=sparseThres-iniSize,targetSparseLevel=sparseRate,targetIter=500)
                activeSet,previousKMEANS=get_activeSet(sparseCand,dataSize=len(train_index),sparseRate=adjRate,previousModel=previousKMEANS,method=sparseUpdate)
                print('adj sparseRate: %f '%(adjRate))
            else:
                activeSet=get_activeSet(sparseCand,dataSize=len(train_index),sparseRate=sparseRate)
        estimator=SGPKMC(rbf_scale=rbf_scale,global_iter=global_iter,gamma_iter=gamma_iter,C=C,kernel=kernel,activeSet=activeSet)
        mc_model=ovr(estimator,n_jobs=njob)
        t1=time.time()
        mc_model.fit(X[train_index,:],Y[train_index])
        t2=time.time()
        score=mc_model.score(X[test_index],Y[test_index])
        t3=time.time()
        if((checkPoison is True) and  (len(ourRes)>5) and ((ourRes[-1]-score)>0.25) ):
                #if the sample poison the model, retract the sample
            poisonSamples.append(train_index[-1])   
            train_index=train_index[0:-1]
            print('Identify a poison sample with %f performance decrease' %(ourRes[-1]-score))#poison data will be put back at the end
        else:
            ourRes.append(score)
            i+=1       
                #exeTime.append(t2-t1)
        sampleIndex, total_mean_var, total_mean_en , max_var= ALSample(X,Y,mc_model,train_index,candidate_index,eta=decayedEta,anchorNum=anchorNum,top=top,noise=noise)
        t4=time.time()
        print('Iter: %d. Score: %f.train time: %f,test time %f, sample time %f' %(i,score,t2-t1,t3-t2,t4-t3))
        var_mean.append(total_mean_var)
        en.append(total_mean_en)
        var_max.append(max_var)
        train_index=train_index+[sampleIndex]
        candidate_index=[x for x in candidate_index if x not in [sampleIndex]]
    return [ourRes,var_mean,en,var_max,train_index,candidate_index,test_index,poisonSamples]
