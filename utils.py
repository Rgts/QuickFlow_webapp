import numpy as np
import pandas as pd
from scipy.spatial import distance


def Read_inp(fn):

    #Read file and store in dataframe
    df = pd.read_csv(fn, header=None, sep='\n')
    df = df[0].str.split(',', expand=True)
    df.columns = ['col_1', 'col_2', 'col_3', 'col_4']

    #Return the index that split nodes and elements (*ELEMENT keyword)
    split_idx = df.query("col_1 == '*ELEMENT'").index.tolist()[0]

    #Extract nodes block
    nodes = df.iloc[:split_idx, :]
    #Clean all non-numeric values
    nodes = nodes[pd.to_numeric(nodes['col_1'], errors='coerce').notnull()]
    #Cast to float
    nodes = nodes.astype(float)
    nodes["col_1"] = nodes["col_1"].astype(int)
    #Set index
    nodes = nodes.set_index('col_1')
    #Cast to numpy
    nodes = nodes.iloc[:, 0:2].to_numpy()

    #Extract elements block
    elements = df.iloc[split_idx + 1:, :]
    #Clean all non-numeric values
    elements = elements.dropna(axis=0, how='any')
    elements = elements[pd.to_numeric(elements['col_1'],
                                      errors='coerce').notnull()]
    #Cast to int
    elements = elements.astype(int)
    #Set index
    elements = elements.set_index('col_1')
    #Cast to numpy
    elements = elements.to_numpy() - 1

    return nodes, elements


def Build_dist_injection_points(idx, nodes):
    #idx=[100,20,3,455]
    target_points = nodes[idx, :]
    dist = np.min(distance.cdist(nodes, target_points, 'euclidean'), axis=1).T
    return dist

def Detect_free_nodes(elements):
    #build edges array
    edges = np.empty(elements.shape, dtype='object')
    for i in range(elements.shape[0]):
        elements[i, :].sort()
        edges[i, 0] = str(elements[i, 0]) + "-" + str(elements[i, 1])
        edges[i, 1] = str(elements[i, 0]) + "-" + str(elements[i, 2])
        edges[i, 2] = str(elements[i, 1]) + "-" + str(elements[i, 2])

    #detect free nodes : edges found once
    values, counts = np.unique(edges.flatten(), return_counts=True)
    free_edges = values[counts == 1]
    free_nodes = []
    for edge in free_edges:
        free_nodes.append(int(edge.split("-")[0]))
        free_nodes.append(int(edge.split("-")[1]))

    #plt.plot(nodes[free_nodes, 0], nodes[free_nodes, 1], 'o', color='black')
    return free_nodes
