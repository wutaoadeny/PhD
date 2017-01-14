#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements SIR model.
"""
import linecache
import string
import os
import math
import time
import random
import networkx as nx
import Centrality as Ranking_methods

'''Implement of SIR model'''
def Optimal_Influence_Spreading(G, Centrality):
    ''' '''
    '''
    SIR model: Susceptible(S):0, Infected(I):1, and Recovered(R):-1;
    Infected probability U, Recovered probability B, Ratio of source spreaders P;
    Infected scale F(t) = number of infected and recovered nodes at time t;
    Node Centrality is used to choose top r spreader.
    '''
    #print "Optimal_Influence_Spreading"
    U = 0.1
    B = 0.05
    P = 0.005
    Gn = G.copy()
    nnodes = Gn.number_of_nodes()
    Source_num =  1#int(nnodes*P)
    #print "Source_num:", Source_num

    Ranking = Ranking_methods.Nodes_Ranking(Gn, Centrality)
    Ranking = sorted(Ranking.iteritems(), key=lambda d:d[1], reverse = True)
    #print "Ranking results for spreading", Ranking

    '''节点状态初始化'''
    Vector_Centrality = dict([(n, 0) for n in Gn])
    for source_num in range(0, Source_num):
        Source_ID = Ranking[source_num][0]
        Vector_Centrality[Source_ID] = 1
        #print "Source_ID:", Source_ID
    #end
    #print Vector_Centrality

    '''从源节点开始进行传播迭代，time = 0'''
    Infected_Num_List = []
    Infected_Num_List.append(Source_num/float(nnodes))
    Active_Measure = 0
    for i in range(200): #最大迭代次数
        #（1）
        Vector_Centrality_Old = Vector_Centrality
        #（2）
        '''infected nodes对susceptible nodes进行传播'''
        status_flag = False
        for node in Vector_Centrality.keys(): ##对第node行（节点node）
            '''infected nodes'''
            if Vector_Centrality_Old[node] == 1:
                '''(1) infected node infects susceptible neighbors with probability U 不进则（退）'''
                for ngb in Gn.neighbors(node):
                    '''susceptible neighbors'''
                    if Vector_Centrality_Old[ngb] == 0:
                        prob = random.random()
                        if prob < U:
                            Vector_Centrality[ngb] = 1
                            status_flag = True
                        #end if
                    #end if
                #end for
                '''(2) infected node recovered with probability B （不进）则退'''
                prob = random.random()
                if prob < B:
                    Vector_Centrality[node] = -1
                #end if
            #end if
        #end for
        if status_flag != True:
            Active_Measure = Active_Measure + 1
        #（3）
        Flag = True
        Current_infected_nodes_num = 0
        #print Vector_Centrality
        for node in Vector_Centrality.keys():
            '''infected and recovered （1， -1）状态的节点数量，即 被传染过的节点数量'''
            if Vector_Centrality[node] != 0:
                Current_infected_nodes_num = Current_infected_nodes_num + 1
            #end if
            '''判断是否还存在infected节点'''
            if Vector_Centrality[node] == 1:
                Flag = False
            #end if
        #end for
        Infected_Num_List.append(Current_infected_nodes_num/float(nnodes))
        #(4)
        '''如果全部节点都是1，-1，即全部节点都被传染过，则停止迭代；   如网络中不再存在infected节点，则停止迭代'''
        if (Current_infected_nodes_num == len(Vector_Centrality.keys())) or (Flag == True) or (Active_Measure == 20):
            print Infected_Num_List
            return Infected_Num_List
    #end for
    return Infected_Num_List


'''基于SIR传播模型的网络节点实际spreading power度量 SIR model based node ranking '''
'''SIR model'''
def Spreading_Scale(G, nid):
    ''' '''
    '''
    SIR model: Susceptible(S):0, Infected(I):1, and Recovered(R):-1;
    Infected probability U, Recovered probability B, Ratio of source spreaders P;
    Infected scale F(t) = number of infected and recovered nodes at time t;
    Node Centrality is used to choose top r spreader.
    '''
    #print "Optimal_Influence_Spreading"
    U = 0.2
    B = 0.5
    Gn = G.copy()

    '''节点状态初始化'''
    Vector_Centrality = dict([(n, 0) for n in Gn])
    Vector_Centrality[nid] = 1
    #print Vector_Centrality

    '''从源节点开始进行传播迭代，time = 0'''
    Active_Measure = 0
    status_flag = False
    for i in range(200): #最大迭代次数
        #（1）
        Vector_Centrality_Old = Vector_Centrality
        #（2）
        '''infected nodes对susceptible nodes进行传播'''
        for node in Vector_Centrality.keys(): ##对第node行（节点node）
            '''infected nodes'''
            if Vector_Centrality_Old[node] == 1:
                '''(1) infected node infects susceptible neighbors with probability U 不进则（退）'''
                for ngb in Gn.neighbors(node):
                    '''susceptible neighbors'''
                    if Vector_Centrality_Old[ngb] == 0:
                        prob = random.random()
                        if prob < U:
                            Vector_Centrality[ngb] = 1
                            status_flag = True
                        #end if
                    #end if
                #end for
                '''(2) infected node recovered with probability B （不进）则退'''
                prob = random.random()
                if prob < B:
                    Vector_Centrality[node] = -1
                #end if
            #end if
        #end for
        if status_flag != True:
            Active_Measure = Active_Measure + 1
        else:
            Active_Measure = 0
            status_flag = False
        #（3）
        Flag = True
        Current_infected_nodes_num = 0
        for node in Vector_Centrality.keys():
            '''infected and recovered （1， -1）状态的节点数量，即 被传染过的节点数量'''
            if Vector_Centrality[node] != 0:
                Current_infected_nodes_num = Current_infected_nodes_num + 1
            #end if
            '''判断是否还存在infected节点'''
            #if Vector_Centrality[node] == 1:
            #    Flag = False
            #end if
        #end for
        '''如果全部节点都是1，-1，即全部节点都被传染过，则停止迭代；   如网络中不再存在infected节点，则停止迭代'''
        if (Current_infected_nodes_num == len(Vector_Centrality.keys())) or (Active_Measure == 10):
            #print Current_infected_nodes_num
            #print nid, Active_Measure
            return Current_infected_nodes_num
    #end for



'''基于SIR传播模型的网络节点实际spreading power度量 SIR model based node ranking '''
def Spreading_Based_Ranking(G):
    ''' '''
    '''
    SIR model: Susceptible(S):0, Infected(I):1, and Recovered(R):-1;
    Infected probability U, Recovered probability B, Ratio of source spreaders P;
    Infected scale F(t) = number of infected and recovered nodes at time t;
    Node Centrality is used to choose top r spreader.
    '''
    #print "SIR model based node ranking"
    Vector_Centrality = dict([(n,  0) for n in G])
    #print G.nodes()
    for nid in G.nodes():
         Vector_Centrality[nid] = Spreading_Scale(G, nid)
    #end for
    return Vector_Centrality


'''基于SIR传播模型的网络节点实际spreading power度量'''
'''Average, multiple conduction of Spreading_Based_Ranking() '''
def SIR_Centrality(G):
    Centrality = dict([(n, 0.0) for n in G])
    for i in range(30):
        Ranking = Spreading_Based_Ranking(G)
        for n in G:
            Centrality[n] = Centrality[n] + Ranking[n]
        #end for
    #end for
    print "Centrality",  sorted(Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return Centrality





