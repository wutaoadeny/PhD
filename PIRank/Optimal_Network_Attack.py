#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements dynamic processes.
"""
import linecache
import string
import os
import math
import time
import networkx as nx
import Centrality as Ranking_methods



#************************************************************************
def Optimal_Percolation_Simultaneous_Attack(G, Centrality):
    #print "Optimal_Percolation_Simultaneous_Attack"
    Gn = G.copy()
    Ranking = Ranking_methods.Nodes_Ranking(Gn, Centrality)
    Ranking = sorted(Ranking.iteritems(), key=lambda d:d[1], reverse = True)

    Giant_Component_Size_List = []
    Component_Num_List = []
    for nid in Ranking:
        G.remove_node(nid[0])
        ### Get the Greatest Component of Networks #####
        Components = sorted(nx.connected_components(G), key = len, reverse=True)
        if len(Components) >= 1:
            Giant_Component_Size = len(Components[0])
            if Giant_Component_Size > 1:
                Giant_Component_Size_List.append(Giant_Component_Size)
                Component_Num_List.append(len(Components))
    #end for
    return Giant_Component_Size_List,Component_Num_List

#************************************************************************

def Optimal_Percolation_Sequence_Attack(G, Centrality, r=0.025):
    print "Optimal_Percolation_Sequence_Attack"
    Step = int(r*G.number_of_nodes())
    print Step
    Gn = G.copy()
    Ranking = Ranking_methods.Nodes_Ranking(Gn, Centrality)
    Ranking = sorted(Ranking.iteritems(), key=lambda d:d[1], reverse = True)
    #print Ranking
    G.remove_node(Ranking[0][0])

    ### Get the Greatest Component of Networks #####
    Giant_Component_Size_List = []
    Components = sorted(nx.connected_components(G), key = len, reverse=True)
    Giant_Component_Size = len(Components[0])
    Giant_Component_Size_List.append(Giant_Component_Size)
    #print "Components[0]:",Components[0]

    while Giant_Component_Size_List[-1] > 2 and Ranking != {}:
        Gn = G.copy()
        Ranking = Ranking_methods.Nodes_Ranking(Gn, Centrality)
        Ranking = sorted(Ranking.iteritems(), key=lambda d:d[1], reverse = True)
        #print Ranking
        if len(Ranking) > Step:
            for i in range(0,Step):
                G.remove_node(Ranking[i][0])
        Components = sorted(nx.connected_components(G), key = len, reverse=True)
        Giant_Component_Size = len(Components[0])
        Giant_Component_Size_List.append(Giant_Component_Size)

        #print "Giant_Component_Size_List, Components[0], Ranking:",Centrality, Giant_Component_Size_List, Components,G.edges(), Ranking
        #print "Sequence_attack:", Centrality, Ranking[0][0]
    #end while

    return Giant_Component_Size_List
#===============================================================================================





