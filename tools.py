# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:53:50 2017
"""
#from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import ndimage
from sklearn import decomposition
import scipy as sp;
import pandas as pd;
import os;
import glob;
import numpy as np;
from sklearn import svm;
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer;
import sklearn.naive_bayes as nb;
import matplotlib.pyplot as plt;
import sklearn.linear_model as ln;
import random; ##to shuffle candidate list in random case
readpath="D:\yourDataPath";
import heapq; ##to get n largest num from a list
import matplotlib.cm as cm;##import a color map
import sklearn.metrics.pairwise as measure;
import numpy as np;
from sklearn.preprocessing import normalize;
from scipy.optimize import minimize;
import multiprocessing as mp
from sklearn.gaussian_process.kernels import RBF;
import time;
#
def classDis(Y):
    #print class distribution
    dic=dict();
    for i in range(len(Y)):
        if(not dic.has_key(Y[i])):
            dic[Y[i]]=1;
        else:
            dic[Y[i]]=dic[Y[i]]+1
    print(dic)
    
#generate copies of train,test and candidate index for different AL
def copyIndex(trainInd,testInd,poolInd):
    return [x for x in trainInd],[y for y in testInd],[z for z in poolInd];


#this removes the digit from dataset
def removeDigits(X,Y):
    #get the index of non-digit Num
    index=[j for j in range(len(Y)) if Y[j] not in [str(i) for i in range(10)]]
    return [X[k] for k in index], [Y[n] for n in index]

def smothCurve(line,fraction=0.20):
    filtered = lowess(line, range(len(line)), is_sorted=True, frac=fraction, it=0);
    x=range(len(filtered[:,0]))
    y=filtered[:,1]
    return x,y;

def testPartition(Y,train_n_candidate=0.66,test=0.33,ntrain=3):
    allIndex=range(len(Y))
    np.random.shuffle(allIndex)
    testIndex=allIndex[0:int(len(allIndex)*test)]
    trainIndex=allIndex[int(len(allIndex)*test):int(len(allIndex)*test)+ntrain]
    poolIndex=allIndex[int(len(allIndex)*test)+ntrain:]       
    return trainIndex,testIndex,poolIndex
        
def genSigni(X):#return the none zero entity count of each datapoint
    signi=[]
    for a in range(X.shape[0]):
        count=0
        for b in X[a,:]:
            if b!=0:
                count+=1
        signi.append(count)
    return np.array(signi)    
#return the index of train, test and pool. make sure at least leastTrain instances are in the training for each class. at least leastPool instances are in the pool. the rest will be in the test
def partitionForAL(Y,leastTrain=1,leastPool=30,randomSeed=0,signi=None):
    trainIndex=[]
    testIndex=[]
    poolIndex=[]
    #if class distribution is balanced, can directly assign how many data instance per class for train/candidate
    if(leastTrain>=1 or leastPool>=1):
        for className in list(set(Y)):
            #get the index of each class
            classIndex=[i for i in range(len(Y)) if Y[i]==className]
            if signi is None:
                np.random.seed(randomSeed)                    
                np.random.shuffle(classIndex)
            else:#if pass a significance list of samples,(how many none zero emptys in the data.)    
                
                classIndex=[classIndex[k] for k in np.argsort(signi[classIndex])]
                #classIndex[np.argsort(signi[classIndex])]
                
            trainIndex=trainIndex+classIndex[0:leastTrain]
            poolIndex=poolIndex+classIndex[leastTrain:leastTrain+leastPool]
            testIndex=testIndex+classIndex[leastTrain+leastPool:len(classIndex)]
        return trainIndex,testIndex,poolIndex
    #for unbalanced class distribution, assign the percentage of instances for each class used for train/candidate    
    if(leastTrain<1 or leastPool<1):
        for className in list(set(Y)):
            #get the index of each class
            classIndex=[i for i in range(len(Y)) if Y[i]==className]    
            np.random.seed(randomSeed)                    
            np.random.shuffle(classIndex)
            trainIndex=trainIndex+classIndex[0:int(np.ceil(leastTrain*len(classIndex)))]
            poolIndex=poolIndex+classIndex[int(np.ceil(leastTrain*len(classIndex))):int(np.ceil((leastTrain+leastPool)*len(classIndex)))]                                             
            testIndex=testIndex+classIndex[int(np.ceil((leastTrain+leastPool)*len(classIndex))):len(classIndex)]                                
        return trainIndex,testIndex,poolIndex
        
        
        
##step wised method to find the optimal candidate
def checkSupportVecs(clf,train_label,train_index):
    for i in range(len(clf.n_support_)):
        support_list=[y for x,y in enumerate(clf.n_support_) if x<=i]
        support_index=reduce(lambda x,y:x+y , support_list)-1
        num_vec=clf.n_support_[i]
        all_support_vec=clf.support_[support_index-num_vec+1:support_index+1]
        print('class '+ str(i) +' :'+str([train_label[train_index[x]] for x in all_support_vec]))
def findAccuDrop(lis):
    res=[]
    for x in range(len(lis)):
        if(x>0):
            if((lis[x]-lis[x-1])<0):
                res.append((x,abs(lis[x]-lis[x-1])*1000))
    return res            

def findBest(data,candidate_index,train_index,train_label,clf,test_data,test_label):
    best_improve=0
    best_index=0
    
    for i in range(len(candidate_index)):
        print('try instance'+str(i))
        train_index.append(candidate_index[i])
        clf.fit(data[0][train_index],[train_label[x] for x in train_index])
        if(clf.score(test_data,test_label)>best_improve):
            best_improve=clf.score(test_data,test_label)
            best_index=candidate_index[i]
    return best_index

##given a candidate datapoint,return  its realclass class's last support vector and the datapoint's sim() is Wrong!for real class is unknown at this point
##use most probabliy class to estimate real class
#note that data here should be the data matrix used to train the model.!!!!!!!!!!!
#the candidate_index_indata and i looks so weird. should use candidate_index_indata=candiate_index[i] instead!!!-----------fix later

def decChange(clf,candidate_index_indata,train_index,train_label,data,mostlikeliclass,i,measureType='cos'):
    def costfun(x):
        return sum((np.dot(A,x)-b)**2)
    def costfun_rbf(x):
        rbf=RBF();
        res=np.dot(np.dot(np.transpose(x),rbf.__call__(S,S)),x)-2*np.dot(np.transpose(x),rbf.__call__(S,b))+rbf.__call__(b,b);
        return res[0][0];
    #realclass=train_label[candidate_index_indata]
    if(measureType=='cos'):
        realclass=mostlikeliclass##use mostlikeliclass to estimate realclass
        ##get the last support vector for most probable
        support_list=[y for x,y in enumerate(clf.n_support_) if x<=realclass]
        support_index=reduce(lambda x,y:x+y , support_list)-1##cause index start from 0
        last_support_vec=clf.support_[support_index]
        num_vec=clf.n_support_[realclass]
        all_support_vec=clf.support_[support_index-num_vec+1:support_index+1]
        #cos=measure.cosine_similarity(data[train_index,:][last_support_vec,:],data[candidate_index_indata,:][[i],:])[0][0]
        cos=0
        cos_sum=measure.cosine_similarity(data[train_index,:][all_support_vec,:],data[candidate_index_indata,:][[i],:])
        cos_sum=[x[0] for x in cos_sum]
        #avg_cos=sum(cos_sum)/len(cos_sum)
        max_cos=max(cos_sum)
        return [max_cos,cos,realclass]
    if(measureType=='covx'):
        realclass=mostlikeliclass##use mostlikeliclass to estimate realclass
        support_list=[y for x,y in enumerate(clf.n_support_) if x<=realclass]
        num_vec=clf.n_support_[realclass]
        support_index=reduce(lambda x,y:x+y , support_list)-1##cause index start from 0
        last_support_vec=clf.support_[support_index]
        all_support_vec=clf.support_[support_index-num_vec+1:support_index+1]
        if(clf.kernel=='rbf'):
            S=data[train_index,:][all_support_vec,:].toarray();
            b=data[candidate_index_indata,:][i,:].toarray();
            print('compute data instance'+str(candidate_index_indata[i]));
            cons = ({'type': 'eq', 'fun': lambda x:  sum(x)-1});
            inix=np.array([0.5 for i in range(S.shape[0])])
            inix=inix[:,np.newaxis];
            bnds = tuple([(0, 1) for i in range(S.shape[0])])
            res = minimize(costfun_rbf, inix,  method='SLSQP', constraints=cons,bounds=bnds)    
        else:    
            A=data[train_index,:][all_support_vec,:].T
            b=data[candidate_index_indata,:][i,:]
            inix=np.array([0.5 for i in range(A.shape[1])])
            cons = ({'type': 'eq', 'fun': lambda x:  sum(x)-1})
            bnds = tuple([(0, 1) for i in range(A.shape[1])])
            res = minimize(costfun, inix,  method='SLSQP', constraints=cons,bounds=bnds)
        #print(res.fun)
        return [res.fun,0,realclass]
    if(measureType=='covx_all'):
        #print('convex_all in desicion change')
        #print('current index:'+ str(i))
        t1=time.time()
        realclass=mostlikeliclass##use mostlikeliclass to estimate realclass
        all_support_vec=clf.support_
        A=data[train_index,:][all_support_vec,:].T
        b=data[candidate_index_indata][i,:]
        inix=np.array([0.5 for k in range(A.shape[1])])
        cons = ({'type': 'eq', 'fun': lambda x:  sum(x)-1})
        bnds = tuple([(0, 1) for k in range(A.shape[1])])
        res = minimize(costfun, inix,  method='SLSQP', constraints=cons,bounds=bnds)
        if(i==0):
            print('time for computing one cadidates error' + str(time.time()-t1))
        '''
        if(A.shape[1]>1):
            for i in range(A.shape[1]-1):
                ##col_i of A=col_i-last_col
                A[:,i]=np.subtract(A[:,i],A[:,A.shape[1]-1])
                ##b=b-last_column
            b=np.subtract(b,A[:,A.shape[1]-1])    
            ##delete last column 
            A=np.delete(A,A.shape[1]-1,1)
        ##covex optimization
        #print(A.shape)
        #print(b.shape)
        opti,rnorm=sp.optimize.nnls(A,b)
        '''
        return [res.fun,0,realclass]



    



#the previous design is a disaster!
#input:
    #data: is the design matrix
    #Y:is the label matrix(one dim)
    #train_index,candidate_index 
def findCandidate(clf,data,train_index,Y,candidate_index,method,lam,gam=0):
    ##for the active labeling
    ##1.train the model on training data
    nc=len(set(Y));
    approximationErrorList=[];    
    def func(a,b,nc,deci,i):
        #lis1=[1.0/2 for x in range(2)];#max entropy given nc
        lis2=[1.0/(nc-2) for x in range(nc-2)];#max entropy given nc
        #maxEn1=sp.stats.entropy(lis1,base=2);  
        maxEn2=sp.stats.entropy(lis2,base=2);
        part1=[a,b];
        #part2=[x for x in deci if (x!=a and x!=b)];      
        
        part1=[x/sum(part1) for x in part1];
        #part2=[x/sum(part2) for x in part2];       
        if(method=='covx'):
           
            represent=decChange(clf,candidate_index,train_index,Y,data,deci.argmax(),i,'covx');
            
            return [(-0.9*sp.stats.entropy(part1,base=2)+(lam)*sp.stats.entropy(deci,base=2)/maxEn2)-gam*represent[0],represent[0]];
        elif (method=='covx_all'):
            represent=decChange(clf,candidate_index,train_index,Y,data,deci.argmax(),i,'covx_all');
            #print(2)
            return [(-0.9*sp.stats.entropy(part1,base=2)+(lam)*sp.stats.entropy(deci,base=2)/maxEn2)-gam*represent[0],represent[0]];
        else:
            return [-0.9*sp.stats.entropy(part1,base=2)+(lam)*sp.stats.entropy(deci,base=2)/maxEn2,0];       
        #return -0.9*sp.stats.entropy([a,b],base=2)+(lam)*sp.stats.entropy([x for x in deci if (x!=a and x!=b)],base=2)/maxEn2;#normalized by max en
    

    clf.fit(data[train_index,:],Y[train_index])
    
    ##2.predict prob on candidate
    dec=clf.predict_proba(data[candidate_index,:])
    resultList=[]
    
    for i in range(dec.shape[0]):
        bVsSb=heapq.nlargest(2, dec[i,:])
        a=max(bVsSb[0],bVsSb[1])
        b=min(bVsSb[0],bVsSb[1])
        if(method=='bvs'):
            
            bVsSb=a-b
            resultList.append(bVsSb)
            
        else:
            #bVsSb=2*a*(a-b)/(2*a-b)
            
            bVsSb=func(a,b,nc,dec[i,:],i)
            approximationErrorList.append(bVsSb[1])    
            resultList.append(bVsSb[0])
  
        
    if(method=='nbvs'):
        targetIndex=resultList.index(min(resultList))
    else:
        targetIndex=resultList.index(min(resultList))
    most_like_class=dec[targetIndex,:].argmax()  
    #print(approximationErrorList[targetIndex])
    if(method=='bvs' or method=='nbvs'):
        return [targetIndex,clf,most_like_class,dec,dec[targetIndex,:],0]    
    else:
        return [targetIndex,clf,most_like_class,dec,dec[targetIndex,:],0]#approximationErrorList[targetIndex]]   


class Convex(object):
    #startWithConAll='True': if you want to start sampling with convex_all method, the error matrix wont be precomputed.  
    def __init__(self,clf,data,train_index,Y,candidate_index,test_index,gam,lam=0,startWithConAll='True',r=0.03):
        #create a errorMat fitst
        #first train the model
        clf.fit(data[train_index,:],Y[train_index]);
        #get the predicted probability
        self.dec=clf.predict_proba(data[candidate_index,:]);
        #create the flag/mask matrix        
        self.mask=np.zeros([len(candidate_index),len(set(Y))]);
        self.clf=clf;
        self.data=data
        self.Y=Y
        self.train_index=list(train_index)
        self.candidate_index=list(candidate_index)
        self.test_index=list(test_index)
        self.nc=len(set(Y))
        self.gam=gam
        self.lam=lam
        self.avoidRecompute=0;#how many errors need to be recompute during the sampling
        self.svschange=[]#how many classes changes their support vecs during every sampling.
        self.al_count=0#record how many data have been sampled. This can affect gam.
        self.currentPerformance=0#the test performance of the current model.
        self.gam_def=gam
        self.r=r
        #compute the candidate-class error matrix
        if(not startWithConAll):
            self.genErrorMat();

    #two purpose, if start sampling with convex method, need to use this method to initialize errorMat
                 #if start sampling with convex_all method, then need to update(recompute) errorMat when switching to convex method.
    def genErrorMat(self):
        self.errorMat=np.zeros([len(self.candidate_index),len(set(self.Y))])
        for i in range(len(self.candidate_index)):
            for j in range(len(set(self.Y))): #number of classes)
                self.errorMat[i,j]=decChange(self.clf,self.candidate_index,self.train_index,self.Y,self.data,j,i,measureType='covx')[0];

        
        
    def getSupportVectorsPerClass(self,clf):
        res=[]
        for clas in range(self.nc):
            realclass=clas##use mostlikeliclass to estimate realclass
            support_list=[y for x,y in enumerate(clf.n_support_) if x<=realclass]
            num_vec=clf.n_support_[realclass]
            support_index=reduce(lambda x,y:x+y , support_list)-1##cause index start from 0
            #last_support_vec=clf.support_[support_index]
            all_support_vec=clf.support_[support_index-num_vec+1:support_index+1]
            res.append(all_support_vec)
        return res;       
    
    def supportVecChange(self,li1,li2):
        #for two lists of support vectors, return the index of the class whose support vector are different
        res=[]
        for i in range(len(li1)):
            if(set(li1[i])!=set(li2[i])):
                res.append(i);
        return res;
    
    #this method should be merged in sample() later
    def EnropySample(self):
        resultList=[]
        for i in range(len(self.candidate_index)):
            entropy=sp.stats.entropy(self.dec[i,:],base=2);
            resultList.append(entropy);
        targetIndex=resultList.index(max(resultList))
        self.train_index.append(self.candidate_index[targetIndex])        
        del self.candidate_index[targetIndex]
        #retrain Model
        self.clf=svm.SVC(decision_function_shape='ovr',C=1.0,probability=True,tol=0.001,kernel='linear');
        self.clf.fit(self.data[self.train_index,:],self.Y[self.train_index]);
        #5 update self.dec
        self.dec=self.clf.predict_proba(self.data[self.candidate_index,:])
    
#eq 6 from paper        
    def updateR(self):
        self.gam=self.gam_def+self.al_count/self.nc*self.r
        #self.lam=self.lam+self.al_count/self.nc*self.r
        
        
    def sample(self,method='convex'):
        self.al_count=self.al_count+1
        #adjust the parameter during the AL process
#        if(self.al_count==1):
#            self.gam=self.gam_def;
#        if(np.mod(self.al_count+1,220)==0):
#            self.gam=self.gam+0.02;
#        
#        if(self.al_count>140 and np.mod(self.al_count,50)==0):
            #self.lam=self.lam-0.02;
#            pass
        self.updateR();
        resultList=[]
        if(np.mod(self.al_count,20)==0):
            print('start to sample the '+str(self.al_count)+' th sample');
            print('current performance: '+ str(self.currentPerformance))
            print('current method: '+method);
            print('current gam: ' + str(self.gam))
        
        if(method=='convex_all'):
            print('sample with convex_all')
            #t1=time.time()
            #for each data point in candidate pool
            for i in range(len(self.candidate_index)):
                #1 compute part1
                bVsSb=heapq.nlargest(2, self.dec[i,:]);
                a=max(bVsSb[0],bVsSb[1])
                b=min(bVsSb[0],bVsSb[1])
                lis2=[1.0/(self.nc-2) for x in range(self.nc-2)];#max entropy given nc
                maxEn2=sp.stats.entropy(lis2,base=2);
                part1=[a,b];
                part1=[x/sum(part1) for x in part1];#'Normalize'
                #compute part2       
                part2=decChange(self.clf,self.candidate_index,self.train_index,self.Y,self.data,0,i,measureType='covx_all')[0];
                #2.5 final score for each candiate
                
                score=(-0.9*sp.stats.entropy(part1,base=2)+(self.lam)*sp.stats.entropy(self.dec[i,:],base=2)/maxEn2)-self.gam*part2           
                resultList.append(score);
                #4 sample the data point and update train and candidate index
            targetIndex=resultList.index(min(resultList))
            self.train_index.append(self.candidate_index[targetIndex])        
            del self.candidate_index[targetIndex]
            #retrain Model
            self.clf=svm.SVC(decision_function_shape='ovr',C=1.0,probability=True,tol=0.001,kernel='linear');
            self.clf.fit(self.data[self.train_index,:],self.Y[self.train_index]);
            #5 update self.dec
            self.dec=self.clf.predict_proba(self.data[self.candidate_index,:])
        else:
            #0 check whether the errorMat has been properly initialized!
            if(not hasattr(self,'errorMat')):
                self.genErrorMat()
            #t1=time.time()
            #for each data point in candidate pool
            for i in range(len(self.candidate_index)):
                #1 compute part1
                bVsSb=heapq.nlargest(2, self.dec[i,:]);
                a=max(bVsSb[0],bVsSb[1])
                b=min(bVsSb[0],bVsSb[1])
                lis2=[1.0/(self.nc-2) for x in range(self.nc-2)];#max entropy given nc
                maxEn2=sp.stats.entropy(lis2,base=2);
                part1=[a,b];
                part1=[x/sum(part1) for x in part1];#'Normalize'
                
                #2 compute part2
                mostLikeClass=self.dec[i,:].argmax()
                if(self.mask[i,mostLikeClass]==0):#support vectors not chaning in the mostLikeClass class
                    self.avoidRecompute=self.avoidRecompute+1;
                    part2=self.errorMat[i,mostLikeClass];
                else:
                    part2=decChange(self.clf,self.candidate_index,self.train_index,self.Y,self.data,mostLikeClass,i,measureType='covx')[0];
                    self.errorMat[i,mostLikeClass]=part2
                    self.mask[i,mostLikeClass]=0
                #2.5 final score for each candiate
                score=(-0.9*sp.stats.entropy(part1,base=2)+(self.lam)*sp.stats.entropy(self.dec[i,:],base=2)/maxEn2)-self.gam*part2           
                resultList.append(score);
            #3 check change of errorMat and update 
                #3.1.first del the row that has been sampled from two matrices
            targetIndex=resultList.index(min(resultList))    
            self.errorMat=np.delete(self.errorMat,(targetIndex),axis=0)
            self.mask=np.delete(self.mask,(targetIndex),axis=0)
                #3.2 get the support vectors (for each class) form the old model
            supportVecs=self.getSupportVectorsPerClass(self.clf)
            #3.3 get the support vectors from the new model(first train the model!)
            self.clf=svm.SVC(decision_function_shape='ovr',C=1.0,probability=True,tol=0.001,kernel='linear');
            self.clf.fit(self.data[self.train_index,:],self.Y[self.train_index]);
            tem_supportVecs=self.getSupportVectorsPerClass(self.clf);
                #3.4 compare two support vecs lists class by class. If the support vector changes in a specific class, update the mask matrix(set the corresponding col to 1)
            compareRes=self.supportVecChange(supportVecs,tem_supportVecs);
                #3.5 update the mask mat
            self.svschange.append(len(compareRes))
            for clas in compareRes:
                self.mask[:,clas]=np.ones(self.mask.shape[0]);
            

        #4 sample the data point and update train and candidate index
        targetIndex=resultList.index(min(resultList))
        self.train_index.append(self.candidate_index[targetIndex])        
        del self.candidate_index[targetIndex]
        #5 update self.dec
        self.dec=self.clf.predict_proba(self.data[self.candidate_index,:])            
            
            #print(str(time.time()-t1)+' used for sampling')
        return None
   
    def evaluate(self):
        self.currentPerformance=self.clf.score(self.data[self.test_index,:],self.Y[self.test_index])
        return  self.currentPerformance;
            

def getYTicks(y,numOfTicks):
    #preserve 3 digits
    yticks=np.around(np.arange(0,np.max(y)+0.001,max(y)/numOfTicks),decimals=3);
    return yticks

def nbvsExample():
    N = 50
    fig=plt.figure();
    a1=fig.add_subplot(121);
    a2=fig.add_subplot(122);
    y1=[0.3/48+abs((float(np.random.rand(1))-0.5)/10) for i in range(50)]
    y1[26]=0.3
    y1[25]=0.4

    y2=[0.25/46+abs((float(np.random.rand(1))-0.5)/10) for i in range(50)]
    y2[24]=0.31
    y2[25]=0.22
    y2[26]=0.14
    y2[27]=0.09

    ticks=[x for x in range(len(range(55))) if np.mod(x+5,5)==0]
    yticks=np.arange(0.0,0.5,0.05)
    a1.bar(range(len(y1)),y1,color='k');
    a2.bar(range(len(y2)),y2,color='k');
    a1.set_title('(a) BvSB',fontsize=30);
    a2.set_title('(b) MC-S',fontsize=30);
    a1.set_xticks(ticks)
    a1.set_xticklabels(ticks,fontsize=25)
    a2.set_xticks(ticks)
    a2.set_xticklabels(ticks,fontsize=25)
    a1.set_yticks(yticks)
    a1.set_yticklabels(yticks,fontsize=25)
    a2.set_yticks(yticks)
    a2.set_yticklabels(yticks,fontsize=25)
    a1.set_xlabel('Classes',fontsize=30)
    a1.set_ylabel('Probability',fontsize=30)
    a2.set_xlabel('Classes',fontsize=30)
    a2.set_ylabel('Probability',fontsize=30)
    plt.show()

        
def dispDataDistribution(bvs,nbvs,dataId,data2,data3):
    width=[0.5 for x in range(len(bvs[dataId]))]
    bvs[dataId].sort();
    nbvs[dataId].sort();
    bvs[data2].sort();
    nbvs[data2].sort();
    bvs[data3].sort();
    nbvs[data3].sort();
    fig=plt.figure();
    ticks=[x for x in range(len(bvs[dataId])+5) if np.mod(x+5,5)==0]
    a1=fig.add_subplot(321);
    a1.bar(range(len(bvs[dataId])),bvs[dataId],width,color='k');
    a2=fig.add_subplot(322);
    a2.bar(range(len(nbvs[dataId])),nbvs[dataId],width,color='k');
    a1.set_title('BvSB sample@'+str(dataId)+' iteration',fontsize=30);
    a2.set_title('MC-S sample@'+str(dataId)+' iteration',fontsize=30);
    a1.set_xlabel('Classes',fontsize=30)
    a1.set_ylabel('Probability',fontsize=30)
    a2.set_xlabel('Classes',fontsize=30)
    a2.set_ylabel('Probability',fontsize=30)
    
    
    a1.set_xticks(ticks)
    a1.set_xticklabels(ticks,fontsize=20)
    a1.set_yticks(getYTicks(bvs[dataId],5))
    a1.set_yticklabels(getYTicks(bvs[dataId],5),fontsize=20)
    
    a2.set_xticks(ticks)
    a2.set_xticklabels(ticks,fontsize=20)
    a2.set_yticks(getYTicks(nbvs[dataId],5))
    a2.set_yticklabels(getYTicks(nbvs[dataId],5),fontsize=20)
    
    a3=fig.add_subplot(323);
    a3.bar(range(len(bvs[data2])),bvs[data2],width,color='k');
    a4=fig.add_subplot(324);
    a4.bar(range(len(nbvs[data2])),nbvs[data2],width,color='k');
    a3.set_title('BvSB sample@'+str(data2)+' iteration',fontsize=30);
    a4.set_title('MC-S sample@'+str(data2)+' iteration',fontsize=30);
    a3.set_xlabel('Classes',fontsize=30)
    a3.set_ylabel('Probability',fontsize=30)
    a4.set_xlabel('Classes',fontsize=30)
    a4.set_ylabel('Probability',fontsize=30)
    a3.set_xticks(ticks)
    a3.set_xticklabels(ticks,fontsize=20)
    
    a3.set_yticks(getYTicks(bvs[data2],5))
    a3.set_yticklabels(getYTicks(bvs[data2],5),fontsize=20)
    a4.set_xticks(ticks)
    a4.set_xticklabels(ticks,fontsize=20)
    a4.set_yticks(getYTicks(nbvs[data2],5))
    a4.set_yticklabels(getYTicks(nbvs[data2],5),fontsize=20)
    
    
    
    
    a5=fig.add_subplot(325);
    a5.bar(range(len(bvs[data3])),bvs[data3],width,color='k');
    a6=fig.add_subplot(326);
    a6.bar(range(len(nbvs[data3])),nbvs[data3],width,color='k');
    a5.set_title('BvSB sample@'+str(data3)+' iteration',fontsize=30);
    a6.set_title('MC-S sample@'+str(data3)+' iteration',fontsize=30);
    a5.set_xlabel('Classes',fontsize=30)
    a5.set_ylabel('Probability',fontsize=30)
    a6.set_xlabel('Classes',fontsize=30)
    a6.set_ylabel('Probability',fontsize=30)
    a5.set_xticks(ticks)
    a5.set_xticklabels(ticks,fontsize=20)
    a5.set_yticks(getYTicks(bvs[data3],5))
    a5.set_yticklabels(getYTicks(bvs[data3],5),fontsize=20)
    a6.set_xticks(ticks)
    a6.set_xticklabels(ticks,fontsize=20)
    a6.set_yticks(getYTicks(nbvs[data3],5))
    a6.set_yticklabels(getYTicks(nbvs[data3],5),fontsize=20)
    plt.show();            

def plot_bar(snapshot):
    x = np.arange(10)
    ys = [i+x+(i*x)**2 for i in range(5)]
    
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    N = 50
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars

    fig, ax = plt.subplots()
    rects=[0 for x in range(5)]
    
    
    for i in range(5):
        
        rects[i] = ax.bar(ind+i*width,snapshot[i+1], width, color=colors[i])
        # add some text for labels, title and axes ticks
        ax.set_ylabel('N_of_instances')
        ax.set_title('First_5*50_iteration')
        ax.set_xticks(ind + width)
        ax.set_xticklabels([str(i) for i in range(50)])
    plt.show()
    
    ##plot each 50 interation
    for i in range(len(snapshot)):
        fig, ax = plt.subplots()
        rects = ax.bar(ind,snapshot[i], width*3)
        ax.set_title(str(i)+'(*50)_iteration')
        ax.set_xticks(ind + width)
        ax.set_ylabel('N_of_instances')
        ax.set_xticklabels([str(a) for a in range(50)])
        print (i)
       ## plt.savefig(dpi=500,filename="F:\Medical\graphs\\"+str(i)+'.png',format='png')
        
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(filename="D:\Medical\graphs\\"+str(i)+'.png',format='png', dpi=300)
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2,stop_words='english') #max_features=n_features,stop_words='english')
    tfidf_vectorizer.decode_error='ignore'  
    tfidf = tfidf_vectorizer.fit_transform(data)
    return [tfidf,tfidf_vectorizer]


def readFile(readpath,dataset=1):
    folder_list=os.listdir(readpath)
    class_label=0
    class_indicator=[]
    text_data=[]
        
    for folder in folder_list:
        
        ##in each folder get file path for all csv files
        if(dataset==2):
            file_path_list=(glob.glob(readpath+folder+'/*.csvmerged.csv'))
        else:
            file_path_list=(glob.glob(readpath+folder+'/*.csv'))
        for csv in file_path_list:
            ##assign each csv with its current class label
            print(csv)
            class_indicator.append(class_label)
            tem_csv=pd.read_csv(csv,header=None) 
            tem_list=tem_csv[2].tolist()
            #print(tem_list)
            temstr=' '.join(tem_list)
            text_data.append(temstr)
            
        class_label+=1
    return [text_data,class_indicator]        
def activeLearn(model,outPut,train_pool,candidate_pool):
    return 0

def covAtK(booleanList):
    res=[]
    for i in range(len(booleanList)):
        res.append(sum(booleanList[0:i+1])*1.0/i)
    return res
    
    
def slideWindow(values,size,step=0.5):
    i=0;
    avg=[];
    while(i+size<len(values)):
        avg.append(np.mean(values[i:i+size]));
        i=int(i+size*step);
    avg.append(np.mean(values[i:len(values)]));    
    return avg;
    
#make all coordinate start from 0    
def offset(X):
    res=[a for a in X];
    for i in range(len(X)):
        minx=np.min(X[i][0]);
        miny=np.min(X[i][1]);        
        if(minx<0):
            res[i][0]=[x-minx for x in X[i][0]]

        if(miny<0):
            res[i][1]=[x-miny for x in X[i][1]]  
    return res
        
#the max coordinate for x and y     
def findMax(X,offset=True):
    maxX=0
    maxY=0
    minX=10000
    minY=10000
    for i in range(len(X)):
        xx=np.max(X[i][0])
        yy=np.max(X[i][1])
        mx=np.min(X[i][0])
        my=np.min(X[i][1])
        if(xx>maxX):
            maxX=xx;
        if(yy>maxY):
            maxY=yy;
        if(mx<minX):
            minX=mx
        if(my<minY):
            minY=my
    print('maxX is:'+str(maxX)+';  MaxY is:'+str(maxY))
    return maxX,maxY        

#assume orginal image is 1500*1500 down size to 200 by 200. The points are dialated 3 times before down sizing    
def toMat(X,dialateIter=3,downSizeTo=200,test=False):
    result=np.ones([len(X),int((downSizeTo)*(downSizeTo))]);
    #get the original shape box:
    maxX,maxY=findMax(X);
    for i in range(len(X)):
    
        img=X[i]
        col=np.array(img[0])
        row=np.array(img[1])
        data=np.ones(len(img[0]))
        
        #create imgMat
        matImg=sp.sparse.csr_matrix((data,(row,col)),shape=(maxY+1,maxX+1)).toarray()
        #dialate image
        matImg=ndimage.morphology.binary_dilation(matImg,iterations=dialateIter);
        if(i==0):#show the example of original image for the first image
            yy=np.where(matImg>0)[0]
            yy=-1*yy
            xx=np.where(matImg>0)[1]
            plt.plot(xx,yy,'o')
        #downSizetheImg stepwisely
        for j in range(downSizeTo,int(max(maxX,maxY)),100)[::-1]:
            matImg=sp.misc.imresize(matImg,(j,j));
        if(i==0):#show the example of downsizing image for the first image
            yy=np.where(matImg>0)[0]
            yy=-1*yy
            xx=np.where(matImg>0)[1]
            plt.plot(xx,yy,'o')
        #reshape the image and adds it to result            
        result[i,:]=matImg.reshape(1,downSizeTo*downSizeTo);
        if(test is True):
            print('this is only test for reshape!')
            break;
    return result;
    
#plot data from sparse mat. Just for test    
def plotMatData(mat,Y,classType='a',index=None,shape=(200,200)):
    if(index is not None):
        img=mat[index,:].todense()
        img=img.reshape(shape[0],shape[1]);
        yy=np.where(img>0)[0]
        yy=-1*yy
        xx=np.where(img>0)[1]
        plt.plot(xx,yy,'o')
    else:
        ind=[i for i in range(len(Y)) if Y[i]==classType]
        nrow=int(np.ceil(np.sqrt(len(ind))))
        fig,ax=plt.subplots(nrows=nrow,ncols=nrow);
        data=mat[ind,:]
        for i in range(nrow):
            for j in range(nrow):
                if((i*nrow+j)<len(ind)):
                    img=data[i*nrow+j,:].todense()
                    img=img.reshape(shape[0],shape[1]);
                    xx=np.where(img>0)[1]
                    yy=np.where(img>0)[0]
                    yy=-1*yy  
                    ax[i][j].plot(xx,yy,'.')
        
    
def dimReduction(X,numOfComponent=range(10,100,20)):
    Var_Exp=[];
    #n_comp = 500
    for i in numOfComponent:
        print(i)
        svd = decomposition.TruncatedSVD(n_components=i, algorithm='arpack')
        svd.fit(X)
        Var_Exp.append(svd.explained_variance_ratio_.sum())
    plt.plot(numOfComponent,Var_Exp)


def shuffle(X,Y,content=None,seed=0):
    rowIndex=range(X.shape[0]);
    #make shuffle predictabel；
    np.random.seed(seed);
    np.random.shuffle(rowIndex);
    X=X[rowIndex,:];
    Y=[Y[i] for i in rowIndex];
    if (content is not None) :
        content=[content[x] for x in rowIndex];
    return X,Y,content;        
    
def interpolant(rawData,numOfPoints=20):
    newData=[]
    #for each image,
    for img in rawData:
        newImg=[[],[]]
        for indx in range(len(img[0])-1):
            #insert numOfPoints points inbetween each pair of points
            if(img[0][indx]<img[0][indx+1]):
                x=np.array([img[0][indx],img[0][indx+1]])
                y=np.array([img[1][indx],img[1][indx+1]])
            else:
                x=np.array([img[0][indx+1],img[0][indx]])
                y=np.array([img[1][indx+1],img[1][indx]])
            pointsX=np.linspace(x[1],x[0],numOfPoints);
            pointsY=np.interp(pointsX, x, y);
            newImg[0]=newImg[0]+pointsX.tolist();
            newImg[1]=newImg[1]+pointsY.tolist();
        newData.append(newImg)
    return newData;
            
        
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#see how many candidate can use lookup convex error table to avoid recomputing the AL score for sampling
def avoidRecompute(convexAL):
    ar=convexAL.avoidRecompute;
    st=len(convexAL.candidate_index)+500 
    end=len(convexAL.candidate_index)
    ini_compute=convexAL.nc*st
    al_total=(st+end)*500/2;
    #aviodRate=ar/al_total
    print('initially need to compute '+str(ini_compute)+' scores')
    print('in AL: '+str(al_total)+' scores need to be computed')
    print('in total '+ str(1-(al_total+ini_compute-ar)*1.0/(al_total))+' percent of the candidate aviod recomputing the convex approximation part of sampling score ')
    
    

    