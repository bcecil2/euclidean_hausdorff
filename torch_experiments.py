import torch
import torch_geometric.datasets as ds
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from EuclideanSimModel import EuclideanSim
from tqdm import tqdm
from eucl_haus import approx_eucl_haus
import matplotlib.pyplot as plt


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


def trainAndValidate(trainSet,valSet,k=1,dim=3):
    buckets = topK(trainSet, k)
    model = EuclideanSim(buckets,dim=dim)
    count = 0
    for x in tqdm(valSet):
       pred = model.similarity(x.pos)
       count += pred == x.y[0]
    return count/len(valSet)

if __name__ == "__main__":
    train,val = getDS("",500)
    print(len(train))
    print(len(val))
    items_list = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    # Using a dictionary comprehension
    class_names = {index: item for index, item in enumerate(items_list)}
    #Data structure for points Data(pos=[4096, 3], y=[1])
    from collections import defaultdict
    classes = defaultdict(list)
    m = 0
    for t in train[:]:
        m = max(m,t.pos.shape[0])
        classes[t.y[0]].append(t.pos)
    acc = trainAndValidate(train, val, 1)
    print(f"Accuracy : {acc}")
    #print(f"Max size: {m}")
    x_11,x_12 = classes[0][4],classes[0][1]
    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x_11[:,0],x_11[:,1],x_11[:,2])
    # plt.show()
    # other_classes = [classes[i][0] for i in range(1,10)]
    #
    # same_class_distance = approx_eucl_haus(x_11,x_12)[0]
    # print(f"Distance for bathtub with itself: {same_class_distance}")
    # times = 0.0
    # from time import perf_counter
    # for i,o in enumerate(other_classes):
    #     tick = perf_counter()
    #     dh = approx_eucl_haus(x_11,o)[0]
    #     tock = perf_counter()
    #     times += tock - tick
    #     print(f"Distance from {class_names[0]} to {class_names[i+1]}: {dh}")
    # print(f"Avg time {times/9}")


