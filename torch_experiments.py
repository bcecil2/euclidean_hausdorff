import torch
import torch_geometric.datasets as ds
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from EuclideanSimModel import EuclideanSim
from tqdm import tqdm

class ToNumpy(object):
    def __init__(self):
        pass
    def __call__(self,x):
        x.pos = x.pos.numpy()
        x.y = x.y.numpy()
        return x
def topK(ds,k):
    from collections import defaultdict
    d = defaultdict(list)
    for x in ds:
        idx = x.y[0]
        if len(d[idx]) < k:
            d[idx].append(x.pos)
    return d

def getDS(ds_name,num_samples):
    sampler = T.Compose([T.SamplePoints(num_samples),ToNumpy()])
    train = ds.ModelNet("./",transform=sampler)
    val = ds.ModelNet("./", transform=sampler,train=False)
    return train,val


def trainAndValidate(trainSet,valSet,grid_size,k,dim):
    params = {"grid_size": grid_size}
    buckets = topK(trainSet, k)
    model = EuclideanSim(buckets, params,dim=dim)
    count = 0
    for x in tqdm(valSet):
       pred = model.similarity(x.pos)
       count += pred == x.y[0]
    return count/len(valSet)

if __name__ == "__main__":
    train,val = getDS("",4096)
    dim = train[0].pos.shape[1]
    grid_size = 2
    buckets = 1
    accuracy = trainAndValidate(train,val[:5],grid_size,buckets,dim)
    print("Accuracy ", accuracy)
