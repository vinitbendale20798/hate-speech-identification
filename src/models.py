# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


import sklearn.externals
import joblib
import psutil
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


def LogRegtrain(a,b):

    vect_transformed_train = a
    train = b
    
    stime = time()
    modelLog = LogisticRegression(C=100,solver='liblinear',max_iter=1000).fit(vect_transformed_train,train['label'])
    joblib.dump(modelLog,'D:/Major/saved_models/savedmodelLog.pkl')
    
    psutilpercent = psutil.virtual_memory()
    
    etime = time()
    
    return psutilpercent,etime-stime



def NBtrain(a,b):

    vect_transformed_train = a
    train = b
    
    stime = time()
    modelNB = MultinomialNB(alpha=1.7).fit(vect_transformed_train,train['label'])
    joblib.dump(modelNB,'D:/Major/saved_models/savedmodelNB.pkl')
    
    psutilpercent = psutil.virtual_memory()
    
    etime = time()
    
    return psutilpercent,etime-stime



def SGDtrain(a,b):

    vect_transformed_train = a
    train = b
    
    stime = time()
    modelSGD = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3).fit(vect_transformed_train,train['label'])
    joblib.dump(modelSGD,'D:/Major/saved_models/savedmodelSGD.pkl')
    
    psutilpercent = psutil.virtual_memory()
    
    etime = time()
    
    return psutilpercent,etime-stime
