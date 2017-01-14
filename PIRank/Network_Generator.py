#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements network dataset.
"""
import linecache
import string
import os
import math
import time
import networkx as nx
import matplotlib.pyplot as plt

#************************************************************************
def Attributes_of_Graph(G):
    print "*Statistic attributes of graphs:"
    print "N", nx.number_of_nodes(G)
    print "M", nx.number_of_edges(G)

    print "C", nx.average_clustering(G)
    #print "<d>", nx.average_shortest_path_length(G)
    print "r", nx.degree_assortativity_coefficient(G)

    degree_list = list(G.degree_iter())
    max_degree = 0
    min_degree = 0
    avg_degree_1 = 0.0
    avg_degree_2 = 0.0
    for node in degree_list:
        avg_degree_1 = avg_degree_1 + node[1]
        avg_degree_2 = avg_degree_2 + node[1]*node[1]
        if node[1] > max_degree:
            max_degree = node[1]
        if node[1] < min_degree:
            min_degree = node[1]
    #end for
    avg_degree = avg_degree_1/len(degree_list)
    avg_degree_square = (avg_degree_2/len(degree_list)) / (avg_degree*avg_degree)
    print "<k>", avg_degree
    print "k_max", max_degree
    print "H (degree heterogeneity)", avg_degree_square
    print "S (average span of degree distribution)", float(max_degree-min_degree)/G.number_of_nodes()



#*******************************************************************
def Degree_distribution(G):
    Nodes = G.nodes()
    Degree_List = []
    Degree_Dic = {}
    for i in Nodes:
        Degree_List.append(G.degree(i))
    Degree_List = sorted(Degree_List, reverse = False)
    Flag = Degree_List[0]
    Count = 1
    for i in Degree_List[1:]:
        if i !=Flag:
            Degree_Dic[Flag] = Count
            Count = 1
            Flag = i
        else:
            Count = Count + 1
    #end for
    #print Degree_Dic
    n = G.number_of_nodes()
    plt.figure(1)
    ax1 = plt.subplot(111)
    plt.sca(ax1)
    #x = list([(i+1) for i in range(0,len(Degree_List))])
    x = sorted(Degree_Dic.keys(), reverse = False)
    y = []
    for i in x:
        y.append(float(Degree_Dic[i])/n)
    #end for
    plt.plot(x, y, "rs-")
    plt.ylabel("Probability")
    plt.xlabel("Degree K")
    plt.title("Degree distribution of networks")
    plt.show()


#************************************************************************
def ER_Generateor(N=1000, M=3000):
    G = nx.gnm_random_graph(N, M)
    #component of the network
    if nx.is_connected(G) == False:
        # Get the Greatest Component of Networks #####
        components = sorted(nx.connected_components(G), key = len, reverse=True)
        print "Component Number of the Generated Network:", len(components)

        for i in range(1, len(components)):
            for node in components[i]:
                G.remove_node(node)
        #end for
        print nx.is_connected(G)
    #endif

    return G

#************************************************************************
def SF_Generateor(N=1000, m=3):
    G = nx.barabasi_albert_graph(N, m)
    #component of the network
    if nx.is_connected(G) == False:
        # Get the Greatest Component of Networks #####
        components = sorted(nx.connected_components(G), key = len, reverse=True)
        print "Component Number of the Generated Network:", len(components)

        for i in range(1, len(components)):
            for node in components[i]:
                G.remove_node(node)
        #end for
        print nx.is_connected(G)
    #endif

    return G

#************************************************************************
def LFR_Community_Generator(fname = 'LFR_4.txt'):
    try:
        fdobj = open(fname,'r')
    except IOError as e:
        print "***file open error:",e
    else:
        G = nx.Graph()
        for i in range(0,12):
            eline = fdobj.readline()
        eline = fdobj.readline()
        while eline:
            line = eline.strip().split()
            G.add_edge(line[0],line[1])
            eline = fdobj.readline()
        #end while
        fdobj.close()
        return G

#************************************************************************
def ReadTxtData(fname):
    'read Data'
    try:
        fdobj = open(fname,'r')
    except IOError as e:
        print "***file open error:",e
    else:
        tepstr = ''
        Result = []
        eline = fdobj.readline()
        while eline:
            line = eline.strip().split()
            #self.G.add_edge(line[0],line[1])
            tep = (line[0],line[1])
            Result.append(tep)
            eline = fdobj.readline()
    #print Result
    return Result

#************************************************************************

def DivideGmlData(fname):
    'read Data'
    #==========Divide dataset========
    fnode = fname + '.node'
    fedge = fname + '.edge'

    try:
        fobj = open(fname,'r')
        fnobj = open(fnode,'w')
        feobj = open(fedge,'w')
    except IOError as e:
        print "***file open error:",e
    else:
        s1 = 'node'
        s2 = 'edge'
        flag = -1
        line = fobj.readline()
        while line:
            if flag == -1:
                if s1 in line:
                    flag = 1
                    fnobj.write(line)
            elif flag == 1:
                if s2 in line:
                    flag = 2
                    feobj.write(line)
                else:
                    fnobj.write(line)
            else:
                feobj.write(line)
            line = fobj.readline()
        #end while
        fobj.close()
        fnobj.close()
        feobj.close()
    #end try
	
#************************************************************************
    
def ReadGmlData(fname):
    'read Data'
    fedge = fname + '.edge'
    try:
        fdobj = open(fedge,'r')
    except IOError as e:
        print "***file open error:",e
    else:
        s = 'source'
        tepstr = ''
        Result = []
        eline = fdobj.readline()
        while eline:
            if s in eline:
                source = eline[11:]
                source = source.strip('\n')
                
                tline = fdobj.readline() 
                target = tline[11:]
                target = target.strip('\n')
                tep = (source,target)
                Result.append(tep)
            eline = fdobj.readline()
    #print 
    return Result
#************************************************************************


if __name__ == '__main__':
    G = LFR_Community_Generator()
    Attributes_of_Graph(G)
