import keras
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from EuclideanSimModel import EuclideanSim
#return 3 tuples of (x,y) for train, val, test
def getMnist(train=0.1,val=0.1,test=0.1, norm=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if norm:
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

def topK(x,y,k):
    from collections import defaultdict
    d = defaultdict(list)
    for x,y in zip(x,y):
        if len(d[y]) < k:
            d[y].append(x)
    return d

def plotImage(X):
    xs,ys = np.nonzero(X)
    plt.scatter(xs,ys)

def im2points(X):
    xs,ys = np.nonzero(X)
    return np.stack([xs,ys],axis=-1)
def normalize(X):
    N = X/np.max(X)
    return N
if __name__ == "__main__":
    train,val,test = getMnist()
    buckets = topK(*train,3)
    params = {"diam":2,"size":10}
    model = EuclideanSim(buckets,params)
    from loss import haus_dist
    A = buckets[1][0]
    B = buckets[1][1]
    #print(np.max(A))
    #print(haus_dist(A, B, diam=True))
    print(model.similarity(A))
    plotImage(A)
    plt.show()


    """
    take train data and bucket based on label
    use top k elements
    for any new data point
     compute its distance from top k
     aggregate i.e max,min average
     assign to bucket with smallest distance
    """