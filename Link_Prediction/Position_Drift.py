__author__ = 'taowu'
#!/user/bin/env python
#coding:utf-8
# -*- coding: utf-8 -*-

import linecache
import string
import os
import math
import time
import networkx as nx
import  numpy as np
import matplotlib.pyplot as plt
from math import log
from numpy import array


def Sum_of_weight(G):
    #nx.number_of_edges(nx.ego_graph(Hub_ego,n,1))
    EdgeList = G.edges(data=True)  #[(0, 1, {}), (1, 2, {}), (2, 3, {})]
    #print EdgeList
    Sum_of_weight = 0.0
    for edge in EdgeList:
        Sum_of_weight = Sum_of_weight + edge[2]['weight'] #weight=string.atof(line[3]),timestamp=string.atof(line[5]
    #end for
    return Sum_of_weight



def Graph_Spatial_Temporal_Dynamics(G, Iteration):

    #Edge = [('0', '2'),('0', '3'),('0', '4'),('0', '5'),('1', '2'),('1', '4'),('1', '7'),('2', '4'),('2', '5'),('2', '6'),('3', '7'),('4', '10'),('5', '7'),('5', '11'),('6', '7'),('6', '11'),('8', '9'),('8', '10'),('8', '11'),('8', '14'),('8', '15'),('9', '12'),('9', '14'),('10', '11'),('10', '12'),('10', '13'),('10', '14'),('11', '13')]
    #G.add_edges_from(Edge)

    ######Central node's position drift model#######
    NSet = G.nodes()

    ####One iteration of Network Position Drift#### May be need Three times!!!!

    for iterate in range(0, Iteration):
        Active_Nodes_Per_Iteration = 0
        for m in NSet:
            #print m
            ######Spatial-temporal influence of neighbors of node 'm'######
            Neighbors = G.neighbors(m)  #one-hop node set
            Hub_ego = nx.ego_graph(G,m,2) #two-hop ego network
            #Hub_ego.remove_node(m) #Central node's Neighbors' network

            #(1) The Spatial_Influence of Central node's Neighbors -- DecorrelationNodesInf[Neighbors]

            #####Neighbors' independent influence in their one-hop ego network##########
            IndependentNeighborInf = {}
            for n in Neighbors:
                IndependentNeighborInf[n] = float(Sum_of_weight(nx.ego_graph(Hub_ego,n,1)))
            #end for

            ###########Connected Neighbors' common influence#################
            DecorrelationNodesInf = {}
            #========Analysis the connection between neighbors================
            ConnectedNeighborGraph = Hub_ego.subgraph(Neighbors)
            ConnectedNeighbors = sorted(nx.connected_components(ConnectedNeighborGraph), key = len, reverse=True)
            #[['11', '8', '13', '14'], ['12'], ['4']]

            #=======Connected Neighbors' Influence===========================
            for ConnectedNodes in ConnectedNeighbors: #[['11', '8', '13', '14'], ['12'], ['4']]
                Num = len(ConnectedNodes)
                if Num > 1: #['11', '8', '13', '14']
                    Connected_Nodes_Attracor = Num*Sum_of_weight(Hub_ego.subgraph(ConnectedNodes))
                    '''
                    Tep_Hub_Ego = Hub_ego.copy()

                    #Node Fusion-----Fusion other nodes to ConnectedNodes[0] in connected neighbor set.
                    for i in range(1, Num):
                        Fusion_Neighbors = Tep_Hub_Ego.neighbors(ConnectedNodes[i]) #????????????
                        for node in Fusion_Neighbors:
                            Tep_Hub_Ego.add_edge(node, ConnectedNodes[0], weight = Tep_Hub_Ego[node][ConnectedNodes[i]]['weight'], timestamp = Tep_Hub_Ego[node][ConnectedNodes[i]]['timestamp']) #????????????
                        Tep_Hub_Ego.remove_node(ConnectedNodes[i])
                    #end for

                    #Calculate the fusioned_node's influence
                    if Tep_Hub_Ego.degree(ConnectedNodes[0]) <= 1:
                        Fusion_Node_Attracor = 0.1
                    elif Tep_Hub_Ego.degree(ConnectedNodes[0]) > 1:
                        #Fusion_Node_Attracor = nx.number_of_edges(nx.ego_graph(Tep_Hub_Ego,ConnectedNodes[0],1)) #????????????
                        Fusion_Node_Attracor = Sum_of_weight(nx.ego_graph(Tep_Hub_Ego,ConnectedNodes[0],1))
                    Total_ConnectedNodes_Inf = float(Fusion_Node_Attracor)*Num
                    '''

                    #Allocate fusioned_node's influence to all connected nodes proportionally with their independent influence
                    Total_ConnectedNodes_Independent_Inf = 0.0
                    for node in ConnectedNodes:
                        Total_ConnectedNodes_Independent_Inf = Total_ConnectedNodes_Independent_Inf + IndependentNeighborInf[node]
                    for node in ConnectedNodes:
                        DecorrelationNodesInf[node] =  Connected_Nodes_Attracor * ( IndependentNeighborInf[node] / Total_ConnectedNodes_Independent_Inf )

                elif Num == 1: #['12']
                    DecorrelationNodesInf[ConnectedNodes[0]] = IndependentNeighborInf[ConnectedNodes[0]]
                else:
                    print "ERROR!"
                #endif
            #end for
            #end #(1) The Spatial_Influence of Central node's Neighbors -- DecorrelationNodesInf[Neighbors]


            #(2) The Temporal_Influence of Central node's Neighbors -- TemporalNodesInf[Neighbors]   G[0][1][weight], G[0][1][timestamp]
            TemporalNodesInf = {}
            Num = len(Neighbors)  #Neighbors = G.neighbors(m)  #one-hop node set
            if Num > 1: #When the central node has more than one neighbor
                timestamp_set = []
                for neg in Neighbors:
                    #print m, neg, G[m][neg]
                    timestamp_set.append(G[m][neg]['timestamp'])
                #end for
                #Logistic distribution for timestamp_set.
                Total_time = 0.0
                Avg_time = 0.0
                Unit_time = 0.0
                for time in timestamp_set:
                    Total_time = Total_time + time
                Avg_time = Total_time/Num
                sorted_timestamp_set = sorted(timestamp_set)

                if sorted_timestamp_set[-1] != sorted_timestamp_set[0]:
                    Unit_time = (sorted_timestamp_set[-1] - sorted_timestamp_set[0]) / (Num - 1)
                    #timestamps standardization to range [0, 1]
                    for neg in Neighbors:
                        TemporalNodesInf[neg] = math.exp(   (G[m][neg]['timestamp'] - Avg_time)/(2*(Unit_time+1))   ) / (   1 + math.exp(    (G[m][neg]['timestamp'] - Avg_time) / (2*(Unit_time+1))   )    )
                    #End for
                else:
                    for neg in Neighbors:
                        TemporalNodesInf[neg] = 1.0
            elif Num == 1: #When the central node has only one neighbor
                TemporalNodesInf[Neighbors[0]] = 1.0
            else:
                print "ERROR!"
            #endif
            #end #(2) The Temporal_Influence of Central node's Neighbors -- TemporalNodesInf[Neighbors]



            #(3) Spatial-temporal influence: Spatial_Temporal_Inf = {}
            Spatial_Temporal_Inf = {}  #neighbors
            MaxNegSpaInf = sorted(DecorrelationNodesInf.values())[-1]
            MaxNegTepInf = sorted(TemporalNodesInf.values())[-1]
            for node in Neighbors:
                Spatial_Temporal_Inf[node] = (DecorrelationNodesInf[node]/MaxNegSpaInf)  *  (TemporalNodesInf[node]/MaxNegTepInf)  #!!!!!!@@@@@~
                #Spatial_Temporal_Inf[node] = (DecorrelationNodesInf[node]/MaxNegSpaInf)  #SP factor
                #Spatial_Temporal_Inf[node] = (TemporalNodesInf[node]/MaxNegTepInf)  #TE factor
            #end for
            #end (3) Spatial-temporal influence


            #(4) Spatial-temporal Network Position Drift: Spatial_Temporal_Inf = {}
            #Spatial_Temporal_Inf[node]
            Sum_Weight = 0
            Sum_Inf = 0
            for node in Neighbors:
                Sum_Weight = Sum_Weight + G[m][node]['weight']
                Sum_Inf = Sum_Inf + Spatial_Temporal_Inf[node]
            #end for

            for node in Neighbors:
                #Delta similarity
                if G[m][node]['weight']/Sum_Weight <  Spatial_Temporal_Inf[node]/Sum_Inf:
                    #Delta_Weight = - G[m][node]['weight'] * ( G[m][node]['weight']/Sum_Weight - Spatial_Temporal_Inf[node]/Sum_Inf )
                    Delta_Weight = - Sum_Weight * ( G[m][node]['weight']/Sum_Weight - Spatial_Temporal_Inf[node]/Sum_Inf )/2.0
                    Active_Nodes_Per_Iteration = Active_Nodes_Per_Iteration + 1
                else:
                    Delta_Weight = 0
                ##Update##
                G[m][node]['weight'] = G[m][node]['weight']  + Delta_Weight
            #end for
            #end (4) Spatial-temporal Network Position Drift

            #Drawcomgraph(Hub_ego)
            #Drawcomgraph(Tep_Hub_Ego)

        #end for

        ####One iteration of Network Position Drift####

        #print "Active_Nodes_at_Iteration:", Active_Nodes_Per_Iteration, iterate
    #End Iteration

    return G
