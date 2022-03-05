#this is some utils functions that are not handle in the web api
import numpy as np
import pandas as pd
from scipy.spatial import distance



def Build_dist(idx, nodes):
    #idx=[100,20,3,455]
    target_points = nodes[idx, :]
    dist = np.min(distance.cdist(nodes, target_points, 'euclidean'), axis=1).T
    return dist
