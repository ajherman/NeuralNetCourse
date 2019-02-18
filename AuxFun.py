# Auxiliary functions for jupyter scripts

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, Button, Layout,BoundedFloatText, Box, Label
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
from scipy.linalg import null_space, pinv
from mpl_toolkits.mplot3d import Axes3D

def networkNodePositions(layer_sizes):
    n_layers = len(layer_sizes)
    node_pos = []
    for j in range(n_layers):
        layer_size = layer_sizes[j]
        node_pos += [((i+1)/(layer_size+1),j) for i in range(layer_size)]
    return node_pos,len(node_pos)

def fullyConnectedEdges(layer_sizes):
    n_layers = len(layer_sizes)
    sep_idx = [0]+list(np.cumsum(layer_sizes))
    edges = []
    for i in range(n_layers-1):
        idx1,idx2,idx3 = sep_idx[i:i+3]
        edges += [(j,k) for j in range(idx1,idx2) for k in range(idx2,idx3)]
    return edges

def drawNet(layers):
    node_positions,n_nodes=networkNodePositions(layers)
    nodes = [node for node in range(n_nodes)]
    edges = fullyConnectedEdges(layers)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.draw_networkx_nodes(G, node_positions, node_size=30, nodelist=[i for i in range(n_nodes)],node_color="blue")
    nx.draw_networkx_edges(G, node_positions, edges, alpha=1.0, width=0.5)
    plt.figure(1)
    plt.axis('off')
    plt.show()
    
def drawClassifierWithNoHiddenLayer(input_size,hidden_size):
    drawNet([input_size,hidden_size,1])

def drawClassifierWithOneHiddenLayer(input_size,hidden_size):
    drawNet([input_size,hidden_size,1])

def drawClassifierWithTwoHiddenLayers(input_size,hidden1_size,hidden2_size):
    drawNet([input_size,hidden1_size,hidden2_size,1])
    
def floatSlider(value,mini,maxi,step,name,continuous=False):
    return widgets.FloatSlider(value=value,min=mini,max=maxi,step=step,description=name,continuous_update=continuous)
   
def intSlider(value,mini,maxi,name,continuous=False):
    return widgets.IntSlider(value=value,min=mini,max=maxi,step=1,description=name,continuous_update=continuous)
 
def sigma(X):
    return 1./(1+np.exp(-X))

def ButtonGrid(m,n,description=''):
    columns = []
    for i in range(n):
        columns.append(VBox([Button(description=str(i+j),layout=Layout(width='50px', height='35px')) for j in range(m)]))
    columns.append(Label(value='\('+description+'\)'))
    return HBox(columns)

def TextBoxGrid(m,n,description=''):
    columns = []
    for i in range(n):
        columns.append(VBox([BoundedFloatText(value=0.0,min=-1.0,max=1.0,step=0.01,description='',layout=Layout(width='65px', height='30px')) for j in range(m)]))
    columns.append(Label(value='\('+description+'\)'))
    return HBox(columns)

def trainNetwork(X,target,n,m,alpha,nitrs):

    # Init network weights
    dim = np.shape(X)[1]
    w1 = np.random.random((dim,n))-0.5
    b1 = np.random.random((1,n))-0.5
    w2 = np.random.random((n,1))-0.5
    b2 = np.random.random((1,1))-0.5

    # Store training data
    w1s = []
    b1s = []
    w2s = []
    b2s = []
    es = [] # Fraction incorrect
    acs = []

    # Train
    for itr in range(nitrs):
        # Forward
        a = np.dot(X,w1)+b1 
        y = sigma(a) # Hidden layer activity
        b = np.dot(y,w2)+b2
        z = sigma(b) # Output activity
        e = np.sum((z-target)**2)/2. # Error
        ac = np.sum(np.abs(z-target)<0.5) # Number correct

        # Store info
        w1s.append(w1.copy())
        b1s.append(b1.copy())
        w2s.append(w2.copy())
        b2s.append(b2.copy())
        es.append(e.copy())
        acs.append(ac.copy())

        # Back
        dz = z - target # Error at z
        db = z*(1-z)*dz # Error at b
        db2 = np.sum(db,axis=0,keepdims=True) # Error at b2
        dw2 = np.dot(y.T,db) # Error at w2
        dy = np.dot(db,w2.T) # Error at y
        da = y*(1-y)*dy # Error at a
        db1 = np.sum(da,axis=0,keepdims=True) # Error at b1
        dw1 = np.dot(X.T,da) # Error at w1

        # Update
        w1 -= alpha*dw1
        b1 -= alpha*db1
        w2 -= alpha*dw2
        b2 -= alpha*db2

    return np.stack(w1s),np.stack(b1s),np.stack(w2s),np.stack(b2s),np.stack(es),np.stack(acs)

