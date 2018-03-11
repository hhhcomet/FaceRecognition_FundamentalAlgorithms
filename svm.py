#ECE539 Mid-term project
#svm.py
#Created by Rong Zhang
import numpy as np
from numpy import linalg as la
import scipy.io as sio
import random as rd
from sklearn import svm
from sklearn.decomposition import PCA

#load data
matfn='YaleB_32x32.mat'
data=sio.loadmat(matfn)
face=np.array(data['fea'])
label=np.array(data['gnd'])

#input p value, create train set and test set
face=face.T
label=label.T
p=10 #change this p value to test different p
train=np.zeros((1024,38*p))
trainLabel=np.zeros(38*p)
test=np.zeros((1024,2414-38*p))
testLabel=np.zeros(2414-38*p)
index=[0]
ins=1
for j in range(len(label[0])):
	if label[0][j]!=ins:
		index.append(j)
		ins+=1
index.append(2415)
randomList=[]
for i in range(len(index)-1):
	randomList.append(rd.sample(range(index[i],index[i+1]-1),p))
randomList=sum(randomList,[])
cnt1=0
cnt2=0
for i in range(len(face[0])):
	if i in randomList:
		train[:,cnt2]=face[:,i]
		trainLabel[cnt2]=label[:,i]
		cnt2+=1
	elif i not in randomList:
		test[:,cnt1]=face[:,i]
		testLabel[cnt1]=label[:,i]
		cnt1+=1
trainLabel=trainLabel.astype(np.int32)
testLabel=testLabel.astype(np.int32)
train=train.T
test=test.T

#PCA process
#by commenting this code block to disable PCA
pca=PCA(n_components=200)
train=pca.fit_transform(train)
test=pca.transform(test)


#create svm classifier, print the score of the test dataset
clf=svm.SVC(kernel='linear') #change the kernel to test different kernel option
clf.fit(train,trainLabel)
print(p,clf.score(test,testLabel))
