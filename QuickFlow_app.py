import hydralit as hy
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import plotly.express as px
#https://github.com/null-jones/streamlit-plotly-events
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import streamlit.components.v1 as components
import matplotlib.animation as animation

app = hy.HydraApp(title='Quick Flow')
#Initialize variables in session_state
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'elements' not in st.session_state:
    st.session_state.elements = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = None
if 'injection_points' not in st.session_state:
    st.session_state.injection_points = []

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


@app.addapp()
def File():
    #hy.info('Hello from app1')
    st.write("Filename :", st.session_state.filename)

    uploadedfile = st.file_uploader("Choose a file", type="inp")
    if uploadedfile is not None:
        st.write("Uploaded file is ", uploadedfile.name)

        st.session_state.nodes, st.session_state.elements = Read_inp(
            uploadedfile)
        st.session_state.filename = uploadedfile.name


@app.addapp()
def View():
    if st.session_state.filename is not None:
        st.subheader("Mesh loaded")

        #fig = plt.figure(figsize=(1,1))
        #plt.triplot(st.session_state.nodes[:, 0], st.session_state.nodes[:, 1],
        #            st.session_state.elements)
        #plt.axis('scaled')
        #fig.set_figheight(4)

        #plt.show()

        fig, axis = plt.subplots(1, figsize=[10,3])
        axis.triplot(st.session_state.nodes[:, 0], st.session_state.nodes[:, 1],
                    st.session_state.elements,"g-",lw=0.2)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        fig.patch.set_alpha(0)
        axis.patch.set_alpha(0)
        fig.set_figheight(4)

        axis.set_aspect('equal', adjustable='datalim')
        for spine in ['top', 'right', 'left', 'bottom']:
            axis.spines[spine].set_visible(False)
        st.write(fig)

    else:
        st.subheader("No mesh loaded")


@app.addapp()
def Edit():
    st.subheader("Injection points")

    fig = go.Figure(data=go.Scatter(x=st.session_state.nodes[:, 0],
                                    y=st.session_state.nodes[:, 1],
                                    mode='markers',
                                    line_color='blue'), )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        width=800,
        height=800,
        paper_bgcolor='rgba(0,0,0,0)',  #transparent
        plot_bgcolor='rgba(0,0,0,0)',  #transparent
        xaxis={
            'visible': False,
            'showticklabels': False
        },
        yaxis={
            'visible': False,
            'showticklabels': False
        })


    selected_points = plotly_events(fig,
                                    click_event=True,
                                    select_event=False,
                                    hover_event=False)

    if selected_points != []:
        new_pointIndex = selected_points[0]['pointIndex']
        st.session_state.injection_points.append(new_pointIndex)
        st.session_state.injection_points=list(
            set(st.session_state.injection_points))
    if st.button('Remove points'):
        st.session_state.injection_points = []
    st.write(pd.DataFrame(st.session_state.injection_points, columns=['Selection']))


@app.addapp()
def Run():
    fig, ax = plt.subplots()
    ax.axis('scaled')

    dist = Build_dist_injection_points(idx=st.session_state.injection_points,
                                       nodes=st.session_state.nodes)

    max_dist = int(np.max(dist))

    def animate(i):
        ax.clear()
        dist = Build_dist_injection_points(
            idx=st.session_state.injection_points,
            nodes=st.session_state.nodes)

        free_nodes = Detect_free_nodes(st.session_state.elements)

        dist_mold = -Build_dist_injection_points(
            idx=free_nodes,
            nodes=st.session_state.nodes)

        dist = dist - 2 * i
        dist = np.maximum(dist, dist_mold)

        ax.tricontour(st.session_state.nodes[:, 0],
                      st.session_state.nodes[:, 1],
                      st.session_state.elements,
                      dist,
                      levels=0)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        return ax

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=int(max_dist / 2),
                                  interval=50,
                                  blit=False)
    plt.show()
    components.html(ani.to_jshtml(), height=1000)


@app.addapp()
def About():
    hy.info('Hello from app 2')

#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()
