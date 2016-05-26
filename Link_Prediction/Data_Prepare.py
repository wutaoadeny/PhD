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


'''
Data_Prepare.Temporal_Statistic_Func() #For the understanding of the networks.
Data_Prepare.Temporal_Split_Func()     #Split to training set and test set.
Data_Prepare.Graph_Statistic_Attributes()  #Get the Statistic_Attributes of the greatest component of the training graph.
'''

###========================================================================================

def Drawcomgraph(G):
    #Draw the graph
    pos=nx.spring_layout(G)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    #colorList = ['SeaGreen','yellow','brown','pink','purple','blue','green','Salmon','red','c','magenta','orange','white','black','y','skyblue','GreenYellow','cyan']#,'aqua'
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_size=150)
    plt.title("Network Community Analysis")
    plt.show()


def Drawsubgraph(HG, DrawGraph):
    #Draw the graph
    #print HG.nodes()
    pos=nx.spring_layout(HG)
    nx.draw_networkx_edges(HG, pos, alpha=0.4)
    #nx.draw_networkx_labels(HG, pos, font_size=10, font_family='sans-serif')
    i = 0
    colorList = ['SeaGreen','yellow','brown','pink','purple','blue','green','Salmon','red','c','magenta','orange','white','black','y','skyblue','GreenYellow','cyan']#,'aqua'
    for key in DrawGraph.keys():
        nx.draw_networkx_nodes(HG, pos, nodelist=DrawGraph[key], node_size=20,node_color=colorList[i%len(colorList)])
        i = i + 1

    plt.title("Network Community Analysis")
    plt.show()

###========================================================================================



'''Temporal_Statistic func'''
def Temporal_Statistic_Func():
    fnid = 'F:/Link_Prediction_Code/Dataset/6-Wireless_contact.txt'
    try:
        fnobj = open(fnid,'r')
        #foobj = open(foid3,'w')
    except IOError as e:
        print "***file open error:",e
    else:
        #Statistics
        Temporal_Dis_Dic = {}
        #eline = fnobj.readline()
        #eline = fnobj.readline()
        eline = fnobj.readline()
        eline = fnobj.readline()
        while eline:
            info = eline.strip('\n').split()
            if len(info) == 4:
                timestamp = info[3]
                #print timestamp
                if timestamp in Temporal_Dis_Dic.keys():
                    Temporal_Dis_Dic[timestamp] = Temporal_Dis_Dic[timestamp] + 1
                else:
                    Temporal_Dis_Dic[timestamp] = 1
                    #create_time = time.mktime(time.strptime(msginfo[1].split()[1],'%Y-%m-%d-%H:%M:%S'))
                eline = fnobj.readline()
            #endif
        #end while
        print Temporal_Dis_Dic

        dict= sorted(Temporal_Dis_Dic.iteritems(), key=lambda d:string.atoi(d[0]))
        print dict



    fnobj.close()
    #foobj.close()

    return 1





def Temporal_Split_Func():
    fnid3 = 'F:/Link_Prediction_Code/Dataset/6-Wireless_contact.txt'

    foid2 = 'F:/Link_Prediction_Code/Dataset/6-Wireless_contact_Train.txt'
    foid3 = 'F:/Link_Prediction_Code/Dataset/6-Wireless_contact_Test.txt'
    try:
        fnobj = open(fnid3,'r')
        foobj1 = open(foid2,'w')
        foobj2 = open(foid3,'w')
    except IOError as e:
        print "***file open error:",e
    else:
        #Statistics
        '''
        #eline = fnobj.readline()
        #eline = fnobj.readline()
        eline = fnobj.readline()
        eline = fnobj.readline()
        while eline:
            info = eline.strip('\n').split()
            if len(info) == 4:
                if 20733 < string.atoi(info[3]) < 246856:
                    foobj1.write(eline)
                elif 246856 <  string.atoi(info[3]) < 364094:
                    foobj2.write(eline)
                eline = fnobj.readline()
            #endif
        #end while
        '''


        G = nx.Graph()
        eline = fnobj.readline()
        eline = fnobj.readline()
        while eline:
            line = eline.strip().split()
            #edge = (line[0],line[1],line[2],line[3])
            #Edge.append(tep)
            if G.has_edge(string.atoi(line[0]),string.atoi(line[1])):
                G[string.atoi(line[0])][string.atoi(line[1])]['weight'] = G[string.atoi(line[0])][string.atoi(line[1])]['weight'] + string.atof(line[2])
                if G[string.atoi(line[0])][string.atoi(line[1])]['timestamp'] > string.atof(line[3]):
                    G[string.atoi(line[0])][string.atoi(line[1])]['timestamp'] = string.atof(line[3])
            else:
                G.add_edge(string.atoi(line[0]),string.atoi(line[1]),weight=string.atof(line[2]),timestamp=string.atof(line[3]))
            eline = fnobj.readline()
        #end while

        Edges_Set = G.edges(data=True)
        print Edges_Set
        Sorted_Edges_Set = sorted(Edges_Set, key=lambda edge: edge[2]['timestamp'])
        print Sorted_Edges_Set

        Top_L = int(len(Sorted_Edges_Set)*0.05)
        print Top_L
        print len(Sorted_Edges_Set[:-(Top_L+1)])
        print len(Sorted_Edges_Set[-Top_L:])
        print nx.is_connected(G)
        print len(Sorted_Edges_Set)
        print len(G.nodes())

    fnobj.close()
    foobj1.close()
    foobj2.close()

    return 1





def Graph_Statistic_Attributes():
    fname1 = 'F:/Link_Prediction_Code/Dataset/1-Topology_Train.txt'
    fname2 = 'F:/Link_Prediction_Code/Dataset/1-Topology_Train_Regular.txt'
    #Get edge from txt type data
    try:
        fdobj = open(fname1,'r')
        fwobj = open(fname2,'w')
    except IOError as e:
        print "***file open error:",e
    else:
        G = nx.Graph()

        eline = fdobj.readline()
        while eline:
            line = eline.strip().split()
            #edge = (line[0],line[1],line[2],line[3])
            #Edge.append(tep)
            if G.has_edge(string.atoi(line[0]),string.atoi(line[1])):
                G[string.atoi(line[0])][string.atoi(line[1])]['weight'] = G[string.atoi(line[0])][string.atoi(line[1])]['weight'] + string.atof(line[2])
                if G[string.atoi(line[0])][string.atoi(line[1])]['timestamp'] > string.atof(line[3]):
                    G[string.atoi(line[0])][string.atoi(line[1])]['timestamp'] = string.atof(line[3])
            else:
                G.add_edge(string.atoi(line[0]),string.atoi(line[1]),weight=string.atof(line[2]),timestamp=string.atof(line[3]))
            eline = fdobj.readline()
        #end while

    #print nx.is_connected(G)
    #print nx.number_connected_components(G)


    ### Get the Greatest Component of Networks #####
    components = sorted(nx.connected_components(G), key = len, reverse=True)
    for i in range(1, len(components)):
        for node in components[i]:
            G.remove_node(node)
    #end for
    print nx.is_connected(G)


    ####Statistic attributes of graphs##
    print "N", nx.number_of_nodes(G)
    print "M", nx.number_of_edges(G)
    print "C", nx.average_clustering(G)
    print "Cw", nx.average_clustering(G, weight='weight')
    print "<d>", nx.average_shortest_path_length(G)
    print "r", nx.degree_assortativity_coefficient(G)
    #print nx.density(G)
    #print nx.transitivity(G)
    degree_list = list(G.degree_iter())
    #print degree_list
    avg_degree_1 = 0.0
    avg_degree_2 = 0.0
    for node in degree_list:
        avg_degree_1 = avg_degree_1 + node[1]
        avg_degree_2 = avg_degree_2 + node[1]*node[1]
    avg_degree = avg_degree_1/len(degree_list)
    avg_degree_square = (avg_degree_2/len(degree_list)) / (avg_degree*avg_degree)
    print "<k>", avg_degree
    print "H", avg_degree_square


    #Regularization weight, timestamp to range(0,1)
    E = G.edges(data=True)
    TimeSet = []
    for i in range(0,len(E)):
        TimeSet.append(E[i][2]['timestamp'])
    min = sorted(TimeSet)[0]
    max = sorted(TimeSet)[-1]
    #print min, max
    for i in range(0, len(E)):
        fwobj.write(str(E[i][0]) + " " + str(E[i][1]) + " " + str(E[i][2]['weight']) + " " + str(math.exp( (-1 / E[i][2]['weight']) ))
                    + " " +  str( E[i][2]['timestamp'] )
                    + " " +  str(  ( E[i][2]['timestamp']-min ) / ( max-min ) )   + '\n' )

    fwobj.close()
    fdobj.close()
    return 1
