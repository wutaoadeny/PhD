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



def resource_allocation_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def predict(u, v):
        return sum(1.0 / G.degree(w) for w in nx.common_neighbors(G, u, v))

    Rank_List = []
    for u, v in ebunch:
        Rank_List.append((u, v, predict(u, v)))
    return Rank_List #((u, v, predict(u, v)) for u, v in ebunch)
    #return ((u, v, predict(u, v)) for u, v in ebunch)


def adamic_adar_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def predict(u, v):
        return sum(1.0 / math.log(G.degree(w))
                   for w in nx.common_neighbors(G, u, v))

    Rank_List = []
    for u, v in ebunch:
        Rank_List.append((u, v, predict(u, v)))
    return Rank_List #((u, v, predict(u, v)) for u, v in ebunch)
    #return ((u, v, predict(u, v)) for u, v in ebunch)


def common_neighbor_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def predict(u, v):
        return len( sorted(nx.common_neighbors(G, u, v) ))

    Rank_List = []
    for u, v in ebunch:
        Rank_List.append((u, v, predict(u, v)))
    return Rank_List #((u, v, predict(u, v)) for u, v in ebunch)
    #return ((u, v, predict(u, v)) for u, v in ebunch)


def local_path_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def predict(u, v):
        #NeighborSet = nx.all_neighbors(G, u)
        #len( sorted(nx.common_neighbors(G, u, v) ))
        paths = list( nx.all_simple_paths(G, source=u, target=v, cutoff=3))
        print paths
        A2 = 0.0
        A3 = 0.0
        for path in paths:
            if len(path) == 3:
                A2 = A2 + 1.0
            elif len(path) == 4:
                A3 = A3 + 1.0
        return  A2 + 0.001 * A3 #Coefficient = 0.001

    Rank_List = []
    for u, v in ebunch:
        Rank_List.append((u, v, predict(u, v)))
    return Rank_List #((u, v, predict(u, v)) for u, v in ebunch)
    #return ((u, v, predict(u, v)) for u, v in ebunch)



def structure_dependent_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    #C = nx.average_clustering(G)
    #d = nx.average_shortest_path_length(G)
    path_range = max(2, math.ceil(nx.average_shortest_path_length(G)))
    #print path_range

    def predict(u, v):
        #NeighborSet = nx.all_neighbors(G, u)
        #len( sorted(nx.common_neighbors(G, u, v) ))
        SD_Index = {}
        #Generate all simple paths in the graph G from source to target,  length <= cutoff .
        paths = list( nx.all_simple_paths(G, source=u, target=v, cutoff = path_range))
        print paths
        for path in paths:
            if SD_Index.has_key( len(path) ):
                SD_Index[len(path)] = SD_Index[len(path)] + 1.0
            else:
                SD_Index[len(path)] = 1.0
        #end for
        print SD_Index

        #Sum up the num of paths with different length
        Coefficient = 0.6
        SD_Value = 0.0
        key_Sequence = list(sorted(SD_Index.keys()))
        for key in key_Sequence:
            if key != 2:
                SD_Value = SD_Value + math.pow(Coefficient, key-2.0) * SD_Index[key]
        #end for
        return  SD_Value #Coefficient = 0.6

    Rank_List = []
    for u, v in ebunch:
        Rank_List.append((u, v, predict(u, v)))
    return Rank_List #((u, v, predict(u, v)) for u, v in ebunch)



##======================================================================##

def node_strength(G, w):
    Sum_of_weight = 0.0
    neighbors = list(nx.all_neighbors(G, w))
    for node in neighbors:
        Sum_of_weight = Sum_of_weight + G[w][node]['weight']
    #end for

    return Sum_of_weight




def weighted_resource_allocation_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def predict(u, v):
        return sum(   (G[w][u]['weight']+G[w][v]['weight']) / node_strength(G, w) for w in nx.common_neighbors(G, u, v))

    Rank_List = []
    for u, v in ebunch:
        Rank_List.append((u, v, predict(u, v)))
    return Rank_List #((u, v, predict(u, v)) for u, v in ebunch)
    #return ((u, v, predict(u, v)) for u, v in ebunch)


def weighted_adamic_adar_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def predict(u, v):
        return sum(  (G[w][u]['weight']+G[w][v]['weight']) / math.log(1.0+node_strength(G, w))
                   for w in nx.common_neighbors(G, u, v))

    Rank_List = []
    for u, v in ebunch:
        Rank_List.append((u, v, predict(u, v)))
    return Rank_List #((u, v, predict(u, v)) for u, v in ebunch)
    #return ((u, v, predict(u, v)) for u, v in ebunch)


def weighted_common_neighbor_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def predict(u, v):
        return sum(   G[w][u]['weight'] + G[w][v]['weight']  for w in nx.common_neighbors(G, u, v) )

    Rank_List = []
    for u, v in ebunch:
        Rank_List.append((u, v, predict(u, v)))
    return Rank_List #((u, v, predict(u, v)) for u, v in ebunch)
    #return ((u, v, predict(u, v)) for u, v in ebunch)





def weighted_local_path_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def predict(u, v):
        #NeighborSet = nx.all_neighbors(G, u)
        #len( sorted(nx.common_neighbors(G, u, v) ))
        paths = list( nx.all_simple_paths(G, source=u, target=v, cutoff=3))
        #print paths
        A2_weight = 0.0
        A3_weight = 0.0
        for path in paths:
            if len(path) == 3:
                for node in range(0, len(path)-1):
                    A2_weight = A2_weight + G[path[node]][path[node+1]]['weight']
            elif len(path) == 4:
                for node in range(0, len(path)-1):
                    A3_weight = A3_weight + G[path[node]][path[node+1]]['weight']

        #value = sum(   G[w][u]['weight'] + G[w][v]['weight']  for w in nx.common_neighbors(G, u, v) )
        return  A2_weight + 0.001 * A3_weight #+value  #Coefficient = 0.001

    Rank_List = []
    for u, v in ebunch:
        Rank_List.append((u, v, predict(u, v)))
    return Rank_List #((u, v, predict(u, v)) for u, v in ebunch)
    #return ((u, v, predict(u, v)) for u, v in ebunch)






'''
def weighted_structure_dependent_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    #C = nx.average_clustering(G)
    #d = nx.average_shortest_path_length(G)
    #path_range = max(2.0, math.ceil( nx.average_shortest_path_length(G) + (1.0 - nx.average_clustering(G)) ))  #math.floor, math.ceil
    #path_range = path_range + math.floor( 1.0/math.sqrt(nx.average_clustering(G)) )
    path_range = max(2.0, math.floor( nx.average_shortest_path_length(G) ) + 1 )

    #path_range = max(2.0, math.ceil( nx.average_shortest_path_length(G)))
    print "path_range:", path_range

    def predict(u, v):
        #NeighborSet = nx.all_neighbors(G, u)
        #len( sorted(nx.common_neighbors(G, u, v) ))
        SD_Index = {}
        #Generate all simple paths in the graph G from source to target,  length <= cutoff .
        paths = list( nx.all_simple_paths(G, source=u, target=v, cutoff = path_range))
        for path in paths:
            Sum_Weight = 0.0
            for node in range(0, len(path)-1):
                Sum_Weight = Sum_Weight + G[path[node]][path[node+1]]['weight']
            if SD_Index.has_key( len(path) ):
                SD_Index[len(path)] = SD_Index[len(path)] + Sum_Weight
            else:
                SD_Index[len(path)] = Sum_Weight
        #end for
        #print SD_Index

        #Sum up the num of paths with different length
        Coefficient = 0.01
        SD_Value = 0.0
        key_Sequence = list(sorted(SD_Index.keys()))
        #print "Path Length:", key_Sequence
        for key in key_Sequence:
            if key != 2:
                SD_Value = SD_Value + math.pow(Coefficient, key-3.0) * SD_Index[key]
        #end for
        return  SD_Value #Coefficient = 0.6

    Rank_List = []
    for u, v in ebunch:
        Rank_List.append((u, v, predict(u, v)))
    return Rank_List #((u, v, predict(u, v)) for u, v in ebunch)
    #return ((u, v, predict(u, v)) for u, v in ebunch)
'''











def revised_weighted_structure_dependent_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def degree_hetero(G):
        degree_list = list(G.degree_iter())
        avg_degree_1 = 0.0
        avg_degree_2 = 0.0
        for node in degree_list:
            avg_degree_1 = avg_degree_1 + node[1]
            avg_degree_2 = avg_degree_2 + node[1]*node[1]
        avg_degree = avg_degree_1/len(degree_list)
        avg_degree_square = (avg_degree_2/len(degree_list)) / (avg_degree*avg_degree)
        return avg_degree_square

    #C = nx.average_clustering(G)
    #d = nx.average_shortest_path_length(G)
    path_range = max(2.0, math.floor( nx.average_shortest_path_length(G) ))

    def predict(u, v):
        shortest_length = 2#nx.shortest_path_length(G,source=u,target=v)
        #print "shortest_length:",shortest_length
        if shortest_length <= path_range:
            paths = list( nx.all_simple_paths(G, source=u, target=v, cutoff = shortest_length+1))
            A2_weight = 0.0
            A3_weight = 0.0
            for path in paths:
                Sum_weight = 0.0
                Strength = 0.0
                if (len(path)-1) == shortest_length:
                    for node in range(0, len(path)-1):
                        Sum_weight = Sum_weight + G[path[node]][path[node+1]]['weight']
                        Strength = Strength + node_strength( G,  path[node] )
                    #end for
                    A2_weight = A2_weight + Sum_weight / ((Strength - node_strength(G,path[0]))/(len(path)-2))#math.sqrt((Strength - node_strength(G,path[0]))/(len(path)-2))
                elif len(path) == shortest_length + 1:
                    for node in range(0, len(path)-1):
                        Sum_weight = Sum_weight + G[path[node]][path[node+1]]['weight']
                        Strength = Strength + node_strength( G,  path[node] )
                    #end for
                    A3_weight = A3_weight + Sum_weight / ((Strength - node_strength(G,path[0]))/(len(path)-2))#math.sqrt(  (Strength - node_strength(G,path[0]))/(len(path)-2)  )
            #end for
            return  A2_weight + 0.001 * A3_weight #+value  #Coefficient = 0.001
        #endif

    '''
    def predict(u, v):
        #NeighborSet = nx.all_neighbors(G, u)
        #len( sorted(nx.common_neighbors(G, u, v) ))
        SD_Index = {}
        #Generate all simple paths in the graph G from source to target,  length <= cutoff .
        range_value = 0
        shortest_length = nx.shortest_path_length(G,source=u,target=v)
        #print "shortest_length:",shortest_length
        #print "shortest_length:",shortest_length
        if shortest_length <= path_range:
            paths = list( nx.all_simple_paths(G, source=u, target=v, cutoff = shortest_length+1))
            A2_weight = 0.0
            A3_weight = 0.0
            for path in paths:
                Sum_weight = 0.0
                Strength = 0.0
                if (len(path)-1) == shortest_length:
                    for node in range(0, len(path)-1):
                        Sum_weight = Sum_weight + G[path[node]][path[node+1]]['weight']
                        Strength = Strength + node_strength( G,  path[node] )
                    #end for
                    A2_weight = A2_weight + Sum_weight#/ ((Strength - node_strength( G,  path[0] ))/(len(path)-2))
                elif len(path) == shortest_length + 1:
                    for node in range(0, len(path)-1):
                        Sum_weight = Sum_weight + G[path[node]][path[node+1]]['weight']
                        Strength = Strength + node_strength( G,  path[node] )
                    #end for
                    A3_weight = A3_weight + Sum_weight#/ math.sqrt(  (Strength - node_strength( G,  path[0]  ))/(len(path)-2)  )

                #print path
                #print Strength, node_strength( G,  path[0] ), len(path)-2
            #endfor
            #print "Common nodes",len(paths),len(list(nx.common_neighbors(G, u, v)))
            #print "WRA",sum(   (G[w][u]['weight']+G[w][v]['weight']) / node_strength(G, w) for w in nx.common_neighbors(G, u, v))
            #print "WSD",A2_weight
            return  A2_weight + 0.001 * A3_weight #+value  #Coefficient = 0.001
        #endif
    '''
    Rank_List = []
    for u, v in ebunch:
        Rank_List.append((u, v, predict(u, v)))

    return Rank_List #((u, v, predict(u, v)) for u, v in ebunch)
    #return ((u, v, predict(u, v)) for u, v in ebunch)





##=============================================================================#


def Link_Prediction(index, G, ebunch=None):
    #G = nx.complete_graph(5)
    if index == "RA":
        Rank_List = resource_allocation_index(G, ebunch)
    if index == "AA":
        Rank_List = adamic_adar_index(G, ebunch)
    if index == "CN":
        Rank_List = common_neighbor_index(G, ebunch)
    if index == "LP":
        Rank_List = local_path_index(G, ebunch)
    if index == "SD":
        Rank_List = structure_dependent_index(G, ebunch)
    return Rank_List






def Wighted_Link_Prediction(index, G, ebunch=None):
    if ebunch != None:
        ebunch = [ (ebunch[0], ebunch[1]) ]
    #end if

    if index == "WRA":
        Rank_List = weighted_resource_allocation_index(G, ebunch)
    if index == "WAA":
        Rank_List = weighted_adamic_adar_index(G, ebunch)
    if index == "WCN":
        Rank_List = weighted_common_neighbor_index(G, ebunch)
    if index == "WLP":
        Rank_List = weighted_local_path_index(G, ebunch)
    if index == "WSD":
        Rank_List = revised_weighted_structure_dependent_index(G, ebunch)
        '''
        preds = weighted_structure_dependent_index(G, ebunch)
        for u, v, p in preds:
            print '(%d, %d) -> %.8f' % (u, v, p)
        '''
    return Rank_List