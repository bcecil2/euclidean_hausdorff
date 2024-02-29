import keras
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from EuclideanSimModel import EuclideanSim
from tqdm import tqdm

#return 3 tuples of (x,y) for train, val, test
def getMnist(train=0.1,val=0.1,test=1,):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = [im2points(x) for x in x_train]
    x_test = [im2points(x) for x in x_test]

    trainSize = len(x_train)
    testSize = len(x_test)

    trainIdx = int(trainSize*train)
    valIdx = int(trainSize*val)
    testIdx = int(testSize*test)


    trainSplit = (x_train[:trainIdx], y_train[:trainIdx])
    valSplit = (x_train[trainIdx:trainIdx+valIdx], y_train[trainIdx:trainIdx+valIdx])
    testSplit = (x_test[:testIdx], y_test[:testIdx])

    return trainSplit,valSplit,testSplit

def testSplits():
    train, val, test = getMnist(1,1,1)
    assert len(train[0]) == 60000 and len(train[1]) == 60000
    assert len(val[0]) == 0 and len(val[1]) == 0
    assert len(test[0]) == 10000 and len(test[1]) == 10000

    train, val, test = getMnist(.7, .1, 1)
    assert len(train[0]) == 42000 and len(train[1]) == 42000
    assert len(val[0]) == 6000 and len(val[1]) == 6000
    assert len(test[0]) == 10000 and len(test[1]) == 10000

    train, val, test = getMnist()
    assert len(train[0]) == 6000 and len(train[1]) == 6000
    assert len(val[0]) == 6000 and len(val[1]) == 6000
    assert len(test[0]) == 1000 and len(test[1]) == 1000

    #make sure last image in train isnt in val
    assert np.all(train[1][-1] != val[1][0])

def topK(xs,ys,k):
    from collections import defaultdict
    d = defaultdict(list)
    for x,y in zip(xs,ys):
        if len(d[y]) < k:
            d[y].append(x)
    return d

def plotImage(X):
    xs,ys = np.nonzero(X)
    plt.scatter(xs,ys)

def im2points(X):
    xs,ys = np.nonzero(X)
    return np.stack([xs,ys],axis=-1)

def trainAndValidate(trainSet,valSet,size,k):
    params = {"grid_size": size}
    buckets = topK(*trainSet, k)
    valSet = list(zip(valSet[0],valSet[1]))
    model = EuclideanSim(buckets, params)
    count = 0
    for x,y in tqdm(valSet):
       pred = model.similarity(x)
       count += pred == y
    return count/len(valSet)

if __name__ == "__main__":
    train,val,test = getMnist(val=0.01)
    print(trainAndValidate(train,val,size=2,k=1))