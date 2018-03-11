#ECE539 Mid-term project
#Eigenface.py
#Created by Rong Zhang
import numpy as np
from numpy import linalg as la
import scipy.io as sio
import random as rd

#load data
matfn='YaleB_32x32.mat'
data=sio.loadmat(matfn)
face=np.array(data['fea'])
label=np.array(data['gnd'])

#input p value, create train set and test set
face=face.T
label=label.T
p=50 #change this value to test different p value

train=np.zeros((1024,38*p))
trainLabel=np.zeros((1,38*p))
test=np.zeros((1024,2414-38*p))
testLabel=np.zeros((1,2414-38*p))
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
		trainLabel[:,cnt2]=label[:,i]
		cnt2+=1
	elif i not in randomList:
		test[:,cnt1]=face[:,i]
		testLabel[:,cnt1]=label[:,i]
		cnt1+=1

#calculate average, difference and eigenvectors
diff=np.zeros((1024,38*p))
ave=np.mean(train,1)
for i in range(len(train[0])):
	diff[:,i]=train[:,i]-ave
btb=np.dot(diff.T,diff)
eigVal,eigVects=la.eig(btb)
eigsortIndex=np.argsort(-eigVal)
eigsortIndex=np.delete(eigsortIndex,[0,1,2])
k=200 #change this value to test different k value
eigsortIndex=eigsortIndex[:k]
eigVects=eigVects[:,eigsortIndex]

cov=np.dot(diff,eigVects)
bbt=np.dot(diff,diff.T)
eigV,eigVe=la.eig(bbt)
ssort=np.argsort(-eigV)
ssort=np.delete(ssort,[0,1,2]) #delete different numbers here to test the effect of number of deletions
ssort=ssort[:k]
eigVe=eigVe[:,ssort]
cov=eigVe

#predict the testset, print the result
result=[]
for i in range(len(test[0])):
	cur=test[:,i]
	wcur=np.dot(cov.T,(cur-ave))
	wval=np.inf
	wres=0
	for j in range(len(train[0])):
		wdata=np.dot(cov.T,diff[:,j])
		if np.sum(np.square(wcur-wdata))<wval:
			wres=j
			wval=np.sum(np.square(wcur-wdata))
	result.append(trainLabel[0,wres])
	print(trainLabel[:,wres],testLabel[:,i])
count=0
for i in range(len(test[0])):
	if result[i]==testLabel[0,i]:
		count+=1
print(p,count/len(test[0]))












