import streamlit as st
import hydralit_components as hc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import plotly.express as px
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import streamlit.components.v1 as components
import matplotlib.animation as animation
import requests
from utils import *

#Initialize variables in session_state
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'elements' not in st.session_state:
    st.session_state.elements = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = None
if 'injection_points' not in st.session_state:
    st.session_state.injection_points = []
if 'free_nodes' not in st.session_state:
    st.session_state.free_nodes = []

over_theme = {
    'txc_inactive': "white",  #'rgba(120,120,120)',#fontcolor unselected menu
    'menu_background': 'black',  #rgba(0,0,0,0)=transparent
    'txc_active': "orange",  # 'rgba(120,120,120)',#fontcolor selected menu
    'option_active': 'rgba(0,0,0,0)'  #=transparent #bubble on active item
}

#make it look nice from the start
st.set_page_config(
    layout='wide',
    initial_sidebar_state='collapsed',
)

# specify the primary menu definition
menu_data = [
    {
        'label': "QuickFlow",
        'submenu': [{
            'id': 'QuickFlow-About',
            'label': "About"
        }]
    },
    {
        'label':
        "File",
        'submenu': [{
            'id': 'File-Open-from-database',
            'label': "Open from database"
        }, {
            'id': 'File-Open-from_user-data',
            'label': "Open from user data"
        }]
    },
    {
        'label':
        "View",
        'submenu': [{
            'id': 'View-Mesh',
            'label': "Mesh"
        }, {
            'id': 'View-Table',
            'label': "Table"
        }]
    },
    {
        'label':
        "Edit",
        'submenu': [{
            'id': 'Edit-Select-injection-points',
            'label': "Select injection points"
        }]
    },
    {
        'label': "Run",
        'submenu': [{
            'id': 'Run-Run-simulation',
            'label': "Run simulation"
        }]
    },
    {
        'label': "Help",
        'submenu': [{
            'id': 'Help-Demo',
            'label': "Demo"
        }]
    },
]

menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    # home_name='Home',
    # login_name='Logout',
    hide_streamlit_markers=False,  #will show the st hamburger
    use_animation=False,  #motion during selection (False avoid weird behaviour)
    sticky_nav=True,  #at the top or not
    sticky_mode='sticky',  #sticky=don't move with scroll or pinned otherwise
)

if menu_id == 'File-Open-from-database':
    st.subheader("Open file from demo database")
    filelist=[]
    import os
    for root, dirs, files in os.walk("Mesh/"):
        for file in files:
            if file.endswith(".inp"):
                filename=os.path.join(root, file)
                filelist.append(filename)
    choice = st.selectbox('Select a file', filelist)

    st.write("Uploaded file is ", choice)
    st.session_state.filename = choice
    uploadedfile=open(choice)
    url = "https://quickflow-api.herokuapp.com/upload_and_read_inp_2D"
    uploadedfile_dict = {'inpfile': uploadedfile}
    params={"remove_unused_component":True}
    response = requests.post(url, files=uploadedfile_dict, params=params)
    st.session_state.nodes = np.array(response.json()['nodes'])
    st.session_state.elements = np.array(response.json()['elements'])
    st.session_state.free_nodes = response.json()['free_nodes']    # st.session_state.nodes, st.session_state.elements = Read_inp(choice)
    #Each new file load we need to remove old selection in case of existing file
    st.session_state.injection_points = []

if menu_id == 'File-Open-from_user-data':
    st.subheader("Open file from demo database")

    uploadedfile = st.file_uploader("Choose a file", type="inp",)
    if uploadedfile is not None:
        st.write("Uploaded file is ", uploadedfile.name)
        st.session_state.filename = uploadedfile.name

        url = "https://quickflow-api.herokuapp.com/upload_and_read_inp_2D"
        uploadedfile_dict = {'inpfile': uploadedfile}
        params={"remove_unused_component":True}
        response = requests.post(url, files=uploadedfile_dict, params=params)
        st.session_state.nodes = np.array(response.json()['nodes'])
        st.session_state.elements = np.array(response.json()['elements'])
        st.session_state.free_nodes = response.json()['free_nodes']

        #Each new file load we need to remove old selection from potential old
        # file
        st.session_state.injection_points = []

if menu_id == "View-Mesh":

    if st.session_state.filename is None:
        st.subheader("No file")

    else:
        st.subheader("Mesh")
        fig, axis = plt.subplots(1, figsize=[10, 3])
        axis.triplot(st.session_state.nodes[:, 0],
                     st.session_state.nodes[:, 1],
                     st.session_state.elements,
                     "g-",
                     lw=0.2)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        fig.patch.set_alpha(0)
        axis.patch.set_alpha(0)
        fig.set_figheight(4)
        axis.set_aspect('equal', adjustable='datalim')
        for spine in ['top', 'right', 'left', 'bottom']:
            axis.spines[spine].set_visible(False)
        st.write(fig)

if menu_id == "View-Table":
    if st.session_state.filename is None:
        st.subheader("No file")

    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Coordinates table")
            st.write(
                pd.DataFrame({
                    "X": st.session_state.nodes[:, 0],
                    "Y": st.session_state.nodes[:, 1],
                }))
        with col2:
            st.subheader("Connectivity table")
            st.write(
                pd.DataFrame({
                    "Node 1": st.session_state.elements[:, 0],
                    "Node 2": st.session_state.elements[:, 1],
                    "Node 3": st.session_state.elements[:, 2]
                }))

if menu_id == "Edit-Select-injection-points":
    if st.session_state.filename is None:
        st.subheader("No file")
    else:

        st.subheader("Select some injection points")

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
            st.session_state.injection_points = list(
                set(st.session_state.injection_points))
        if st.button('Remove points'):
            st.session_state.injection_points = []
        st.write(
            pd.DataFrame(st.session_state.injection_points,
                         columns=['Selection']))

if menu_id == "QuickFlow-About":
    st.subheader("About QuickFlow")
    st.write("This work is an [open source project]\
                (https://github.com/Rgts/QuickFlow_webapp) hosted on GitHub."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            )
    st.write("Author : [renaud.gantois@gmailcom]\
        (mailto:renaud.gantois@gmail.com)."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    )

if menu_id == "Run-Run-simulation":
    if st.session_state.filename is None:
        st.subheader("No file")
    elif st.session_state.injection_points == []:
        st.subheader("No injection point")

    else : st.subheader("Run flow pattern simulation ?")

    if st.button('Run'):
        #st.write('Why hello there')
        fig, ax = plt.subplots()
        ax.axis('scaled')
        dist = Build_dist(
            idx=st.session_state.injection_points,
            nodes=st.session_state.nodes)
        max_dist = int(np.max(dist))

        def animate(i):
            ax.clear()
            dist = Build_dist(
                idx=st.session_state.injection_points,
                nodes=st.session_state.nodes)

            #free_nodes = Detect_free_nodes(st.session_state.elements)
            dist_mold = -Build_dist(
                idx=st.session_state.free_nodes, nodes=st.session_state.nodes)
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

if menu_id == "Help-Demo":
    st.subheader("Under construction...")
