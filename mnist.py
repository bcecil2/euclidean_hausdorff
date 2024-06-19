import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist
from eucl_haus import approx_eucl_haus

#return 3 tuples of (x,y) for train, val, test
def getMnist(train=0.1,val=0.1,test=1,):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    plotImage(x_train[0])
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

def getMnistFull():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    xs = [im2points(x) for x in x_train]
    xs += [im2points(x) for x in x_test]
    return xs,np.append(y_train,y_test)


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

def plotImage(X):
    xs,ys = np.nonzero(X)
    plt.scatter(xs,ys)

def im2points(X):
    xs,ys = np.nonzero(X)
    points = np.stack([xs,ys],axis=-1).astype(np.float64)
    points /= np.max(cdist(points,points))
    return points

if __name__ == "__main__":
    #train,val,test = getMnist(val=0.01)
    xs,ys = getMnistFull()
    from time import perf_counter
    tick = perf_counter()
    approx_eucl_haus(xs[0], xs[1])
    tock = perf_counter()
    print(tock-tick)
    acc = 0.0
    i = 0
    for x in tqdm(xs):
        dists = []
        for ox in xs:
            dists.append(approx_eucl_haus(x,ox)[0])
        dists[i] = np.inf
        idx = np.argmin(dists)
        acc += ys[idx] == ys[i]
        i += 1
    print(acc/len(xs))
