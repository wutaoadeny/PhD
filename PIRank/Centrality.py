#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements community detection.
"""
import linecache
import string
import os
import math
import time
import networkx as nx
import random

#****************************************************************************************
#Katz_Centrality = nx.katz_centrality(G)
#print "Katz_Centrality:", sorted(Katz_Centrality.iteritems(), key=lambda d:d[1], reverse = True)

def Degree_Centrality(G):
    Degree_Centrality = nx.degree_centrality(G)
    #print "Degree_Centrality:", sorted(Degree_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return Degree_Centrality

def Between_Centrality(G):
    Bet_Centrality = nx.betweenness_centrality(G)
    #print "Bet_Centrality:", sorted(Bet_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return Bet_Centrality

def Closeness_Centrality(G):
    Closeness_Centrality = nx.closeness_centrality(G)
    #print "Closeness_Centrality:", sorted(Closeness_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return Closeness_Centrality

def Page_Rank(G):
    PageRank_Centrality = nx.pagerank(G, alpha=0.85)
    #print "PageRank_Centrality:", sorted(PageRank_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return PageRank_Centrality

def Eigen_Centrality(G):
    Eigen_Centrality = nx.eigenvector_centrality(G)
    #print "Eigen_Centrality:", sorted(Eigen_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return Eigen_Centrality


#**********************************************************************************
def KShell_Centrality(G):
    #网络的kshell中心性
    #The k-core is found by recursively pruning nodes with degrees less than k.
    #The k-shell is the subgraph of nodes in the k-core but not in the (k+1)-core.
    nodes = {}
    core_number = nx.core_number(G) #The core number of a node is the largest value k of a k-core containing that node.
    for k in list(set(core_number.values())):
        nodes[k] = list(n for n in core_number if core_number[n]==k)
    #print core_number #{'1': 2, '0': 2, '3': 2, '2': 2, '4': 1}字典（节点：KShell值）
    #print nodes.keys(),nodes
    KShell_Centrality = core_number
    return KShell_Centrality


#****************************************************************
def Collective_Influence(G, l=2):
    Collective_Influence_Dic = {}
    node_set = G.nodes()
    for nid in node_set:
        CI = 0
        neighbor_set = []
        neighbor_hop_1 = G.neighbors(nid)
        neighbor_hop_2 = []
        for nnid in neighbor_hop_1:
            neighbor_hop_2  = list(set(neighbor_hop_2).union(set(G.neighbors(nnid))))
            #print '2_hop:', nnid, G.neighbors(nnid)
        #end for

        center = [nid]
        neighbor_set = list(   set(neighbor_hop_2).difference(   set(neighbor_hop_1).union(set(center))  )    )
        #print nid, neighbor_hop_1, neighbor_hop_2, neighbor_set

        total_reduced_degree = 0
        for id in neighbor_set:
            total_reduced_degree = total_reduced_degree + (G.degree(id)-1.0)
        #end

        CI = (G.degree(nid)-1.0) * total_reduced_degree
        Collective_Influence_Dic[nid] = CI
    #end for
    #print "Collective_Influence_Dic:",sorted(Collective_Influence_Dic.iteritems(), key=lambda d:d[1], reverse = True)

    return Collective_Influence_Dic

#**********************************************************************************

def Enhanced_Collective_Influence(G, d=2):
    #强化的Collective Influence, 参数d为考虑的范围radius。
    Enhanced_Collective_Influence_Dic = {}

    node_set = G.nodes()
    #对于网路中每个节点。
    for nid in node_set:
        neighbor_hop_1 = G.neighbors(nid)
        neighbor_hop_2 = []
        for ngb1 in neighbor_hop_1:
            neighbor_hop_2 = list(set(neighbor_hop_2).union(G.neighbors(ngb1)))
        #end for
        neighbor_hop_2 = list(  set(neighbor_hop_2).difference( set(neighbor_hop_1).union(set([nid]))  ) )

        #(1)计算Collective_Influence取值
        Total_Reduced_Degree = 0.0
        for id in neighbor_hop_2:
            Total_Reduced_Degree = Total_Reduced_Degree + (G.degree(id)-1.0)
        #end
        Collective_Influence = (G.degree(nid)-1.0) * Total_Reduced_Degree

        #(2)对nid的Collective_Influence进行关于neighbors的Correlation_Intensity强化

        Correlation_Intensity = 0.0

        for id1 in neighbor_hop_2: #Center_set：离中心源点不同层的节点集合
            for id2 in neighbor_hop_2:
                if id1 != id2:
                    Correlation_Intensity = Correlation_Intensity + float(len(set(G.neighbors(id1)).intersection(set(G.neighbors(id2))))) / float(len(set(G.neighbors(id1)).union(set(G.neighbors(id2)))))
        #end for

        Correlation_Intensity_1 = 0.0
        for id1 in neighbor_hop_1: #Center_set：离中心源点不同层的节点集合
            for id2 in neighbor_hop_1:
                if id1 != id2:
                    Correlation_Intensity_1 = Correlation_Intensity_1 + float(len(set(G.neighbors(id1)).intersection( set(G.neighbors(id2)).difference(set([nid]))  ))) / float(len(set(G.neighbors(id1)).union(  set(G.neighbors(id2)).difference(set([nid]))   )))
        #end for
        Correlation_Intensity = 0.5*Correlation_Intensity + Correlation_Intensity_1

        '''
        #SubG_1 = G.subgraph(neighbor_hop_1).copy() #子图
        SubG_2 = G.subgraph(neighbor_hop_2).copy() #子图
        #SubEdge_1 = SubG_1.number_of_edges()
        SubEdge_2 = SubG_2.number_of_edges()
        #SubDegree_1 = sum(G.degree(v) for v in SubG_1.nodes())
        SubDegree_2 = sum(G.degree(v) for v in SubG_2.nodes())
        #Correlation_Intensity = 2*float(SubEdge_1)/(SubDegree_1+1) + float(SubEdge_2)/(SubDegree_2+1)
        Correlation_Intensity = Correlation_Intensity + float(SubEdge_2)/(SubDegree_2+1)
        '''

        #(3)计算邻居结构的均衡性-structural entropy
        #邻居节点的度概率-Degree proporational list
        Degree_List = []
        Total_Degree = 0
        for node in G.neighbors(nid):
            Degree_List.append(G.degree(node))
            Total_Degree = Total_Degree + G.degree(node)
        #end for
        for i in range(0,len(Degree_List)):
            Degree_List[i] = Degree_List[i]/float(Total_Degree)
        #end for
        #计算正则化熵
        Entropy = 0.0
        for i in range(0, len(Degree_List)):
            Entropy = Entropy + ( - Degree_List[i] * math.log( Degree_List[i] ) )
        Entropy = Entropy / math.log( G.degree(nid) + 0.1 )
        #end for

        #（4）计算Enhanced_Collective_Influence(ECI)
        Enhanced_Collective_Influence_Dic[nid] = Collective_Influence * Entropy/(1+Correlation_Intensity)

    #end for

    #print sorted(Enhanced_Collective_Influence_Dic.iteritems(), key=lambda d:d[1], reverse = True)
    return Enhanced_Collective_Influence_Dic





###*****************************************************************************
def MD_Eigen_Centrality_Andy(G):
    #Written by Tao Wu at 2016.10.22.
    #The implementation of "Power Iteration" method for "Leading Eigenvector" of matrix combining with mass diffusion process.
    #R(t + 1) = W · R(t), where W = A ⊙ P, P reflects the effect of mass diffusion process.
    tol=1.0e-3
    nnodes = G.number_of_nodes()
    Vector_Centrality = dict([(n, random.random()) for n in G]) #1.0/len(G) random.random()
    for i in range(300):
        Vector_Centrality_Old = Vector_Centrality
        Vector_Centrality = dict.fromkeys(Vector_Centrality_Old, 0)
        #print Vector_Centrality, Vector_Centrality_Old

        #(1) Calculate the matrix-by-vector product b' = Ab; #R(t + 1) = W · R(t)
        for node in Vector_Centrality: ##对第node行（节点node）[nid; 0]
            for ngb in G:
                if G.has_edge(node, ngb):
                    #print node, ngb, int(G.has_edge(node, ngb))
                    #Vector_Centrality[node] += (1.0/float(G.degree(ngb))) * int(G.has_edge(node, ngb)) * Vector_Centrality_Old[ngb] #W = A ⊙ P
                    #Vector_Centrality[node] += (1.0/math.sqrt(float(G.degree(ngb))) ) * int(G.has_edge(node, ngb)) * Vector_Centrality_Old[ngb] #W = A ⊙ P
                    Vector_Centrality[node] += (1.0/math.pow(float(G.degree(ngb)), 1.0) ) * Vector_Centrality_Old[ngb]

        #(2) Normalize vector: calculate the length of the resultant vector
        try:
            s = 1.0/math.sqrt(sum(v**2 for v in Vector_Centrality.values()))
        # this should never be zero?
        except ZeroDivisionError:
            s = 1.0
        for n in Vector_Centrality:
            Vector_Centrality[n] *= s

        #(3) Check convergence
        err = sum([abs(Vector_Centrality[n] - Vector_Centrality_Old[n]) for n in Vector_Centrality])
        #print "Iteration:",i, sorted(Vector_Centrality_Old.iteritems(), key=lambda d:d[1], reverse = True)

        if err < nnodes*tol:
            return Vector_Centrality
    #end for


###*****************************************************************************
def HC_Eigen_Centrality_Andy(G):
    #Written by Tao Wu at 2016.10.22.
    #The implementation of "Power Iteration" method for "Leading Eigenvector" of matrix combining with heat conduction process.
    #R(t + 1) = W · R(t), where W = A ⊙ P, P reflects the effect of heat conduction process.
    tol=1.0e-2
    nnodes = G.number_of_nodes()
    Vector_Centrality = dict([(n, random.random()) for n in G])#random.random()
    for i in range(100):
        Vector_Centrality_Old = Vector_Centrality
        Vector_Centrality = dict.fromkeys(Vector_Centrality_Old, 0)
        #print sorted(Vector_Centrality_Old.iteritems(), key=lambda d:d[1], reverse = True)

        #(1) Calculate the matrix-by-vector product b' = Ab; #R(t + 1) = W · R(t)
        for node in Vector_Centrality: ##对第node行（节点node）[nid; 0]
            for ngb in G:
                if G.has_edge(node, ngb):
                    #print node, ngb, int(G.has_edge(node, ngb))
                    #Vector_Centrality[node] += (1.0/math.sqrt(float(G.degree(node))) ) * int(G.has_edge(node, ngb)) * Vector_Centrality_Old[ngb] #W = A ⊙ P
                    Vector_Centrality[node] += (1.0/math.pow(float(G.degree(node)), 1.0) )  * Vector_Centrality_Old[ngb]

        #(2) Normalize vector: calculate the length of the resultant vector
        try:
            s = 1.0/math.sqrt(sum(v**2 for v in Vector_Centrality.values()))
        # this should never be zero?
        except ZeroDivisionError:
            s = 1.0
        for n in Vector_Centrality:
            Vector_Centrality[n] *= s

        #(3) Check convergence
        err = sum([abs(Vector_Centrality[n] - Vector_Centrality_Old[n]) for n in Vector_Centrality])
        #print "Iteration:",i, sorted(Vector_Centrality_Old.iteritems(), key=lambda d:d[1], reverse = True)

        if err < nnodes*tol:
            return Vector_Centrality
    #end for






###*****************************************************************************
def Hybrid_Diffusion_Centrality(G):
    #Written by Tao Wu at 2016.10.22.
    #The implementation of "Power Iteration" method for "Leading Eigenvector" of matrix combining with heat conduction & mass diffusion.
    #R(t + 1) = W · R(t), where W = A ⊙ P.
    tol=1.0e-2
    parameter = 0.7
    nnodes = G.number_of_nodes()
    Vector_Centrality = dict([(n,  random.random()) for n in G])
    for i in range(100):
        Vector_Centrality_Old = Vector_Centrality
        Vector_Centrality = dict.fromkeys(Vector_Centrality_Old, 0)
        #print Vector_Centrality, Vector_Centrality_Old

        # do the multiplication y^T = x^T A
        for node in Vector_Centrality:#对第node行（节点node）
            for ngb in G:
                #print node, ngb, int(G.has_edge(node, ngb))
                if G.has_edge(node, ngb):
                    #heat conduction & mass diffusion
                    p = 1.0 / ( math.pow(math.sqrt(G.degree(node)), parameter) * math.pow(math.sqrt(G.degree(ngb)), (1-parameter)))
                    Vector_Centrality[node] += p * Vector_Centrality_Old[ngb]
        #end for
        # normalize vector
        try:
            s = 1.0/math.sqrt(sum(v**2 for v in Vector_Centrality.values()))
        # this should never be zero?
        except ZeroDivisionError:
            s = 1.0
        for n in Vector_Centrality:
            Vector_Centrality[n] *= s

        # check convergence
        err = sum([abs(Vector_Centrality[n] - Vector_Centrality_Old[n]) for n in Vector_Centrality])
        print "Iteration:",i, sorted(Vector_Centrality_Old.iteritems(), key=lambda d:d[1], reverse = True)

        if err < nnodes*tol:
            return Vector_Centrality
    #end for
    #Standard: [('1', 0.5466310265698511), ('2', 0.5171636983834933), ('10', 0.28777901241759757), ('5', 0.269332314258767), ('4', 0.269332314258767), ('9', 0.269332314258767), ('8', 0.269332314258767), ('3', 0.13839646718655957), ('7', 0.13839646718655957), ('6', 0.1309358470722074), ('11', 0.07285964283167869)]
    #Hybrid_Diffusion_Centrality: [('2', 0.3709191922868913), ('1', 0.365432983332083), ('5', 0.346208816860064), ('4', 0.346208816860064), ('9', 0.346208816860064), ('8', 0.346208816860064), ('10', 0.31636930518482415), ('3', 0.22148467732608249), ('7', 0.22148467732608249), ('6', 0.20474837340193858), ('11', 0.09650466178027978)]


###******************************************************************************

def Eigen_Centrality_Andy(G):
    #Written by Tao Wu at 2016.9.28.
    #The implementation of "Power Iteration" method for "Leading Eigenvector" of matrix.
    #R(t + 1) = W · R(t), where W = A ⊙ P.
    #print "Eigen_Centrality_Andy"
    tol=1.0e-2
    nnodes = G.number_of_nodes()
    Vector_Centrality = dict([(n, random.random()) for n in G])
    for i in range(200):
        Vector_Centrality_Old = Vector_Centrality
        Vector_Centrality = dict.fromkeys(Vector_Centrality_Old, 0)
        #print Vector_Centrality, Vector_Centrality_Old

        # do the multiplication y^T = x^T A
        for node in Vector_Centrality: ##对第node行（节点node）
            for ngb in G:
                if G.has_edge(node, ngb):
                    #print node, ngb, int(G.has_edge(node, ngb))
                    Vector_Centrality[node] += Vector_Centrality_Old[ngb] #int(G.has_edge(node, ngb)) *

        # normalize vector
        try:
            s = 1.0/math.sqrt(sum(v**2 for v in Vector_Centrality.values()))
        # this should never be zero?
        except ZeroDivisionError:
            s = 1.0
        for n in Vector_Centrality:
            Vector_Centrality[n] *= s

        # check convergence
        err = sum([abs(Vector_Centrality[n] - Vector_Centrality_Old[n]) for n in Vector_Centrality])
        #print "Iteration:",i, sorted(Vector_Centrality_Old.iteritems(), key=lambda d:d[1], reverse = True)

        if err < nnodes*tol:
            return Vector_Centrality
    #end for
    print "Eigen_Centrality_Andy Iteration Error", err
    print sorted(Vector_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    print sorted(Vector_Centrality_Old.iteritems(), key=lambda d:d[1], reverse = True)



def Eigen_Centrality_Avg(G):
    Eigen_Centrality = dict([(n, 0.0) for n in G])
    for i in range(10):
        Eigen_Ranking = Eigen_Centrality_Andy(G)
        for n in G:
            Eigen_Centrality[n] = Eigen_Centrality[n] + Eigen_Ranking[n]
        #end for
    #end for
    #print "Eigen_Centrality",  sorted(Eigen_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return Eigen_Centrality





###*****************************************************************************
def Weighted_Hybrid_Diffusion_Centrality(G, parameter = 0.2):
    #print "Weighted_Hybrid_Diffusion_Centrality"
    #Written by Tao Wu at 2016.10.22.
    #The implementation of "Power Iteration" method for "Leading Eigenvector" of matrix combining with heat conduction & mass diffusion.
    #R(t + 1) = W · R(t), where W = A ⊙ P ⊙ S.
    tol=1.0e-2
    nnodes = G.number_of_nodes()
    Vector_Centrality = dict([(n, random.random()) for n in G])
    for i in range(100):
        Vector_Centrality_Old = Vector_Centrality
        Vector_Centrality = dict.fromkeys(Vector_Centrality_Old, 0)
        #print Vector_Centrality, Vector_Centrality_Old

        # do the multiplication y^T = x^T A
        for node in Vector_Centrality:#对第node行（节点node）
            for ngb in G:
                #print node, ngb, int(G.has_edge(node, ngb))
                if G.has_edge(node, ngb):
                    #heat conduction & mass diffusion
                    #Corre_Score = float(len(  set(G.neighbors(node)).intersection( set(G.neighbors(ngb))  ))) / float(len( set(G.neighbors(node)).union( set(G.neighbors(ngb)) )))
                    #p = (1.0) / ( math.pow(G.degree(node),parameter) * math.pow(G.degree(ngb),(1-parameter))) #(1.0 - Corre_Score)
                    #Vector_Centrality[node] +=  p * Vector_Centrality_Old[ngb]
                    Vector_Centrality[node] +=  (1.0/ (  math.pow(G.degree(node), parameter) * math.pow(G.degree(ngb), (1-parameter)) )   ) * Vector_Centrality_Old[ngb]
        #end for
        # normalize vector
        try:
            s = 1.0/math.sqrt(sum(v**2 for v in Vector_Centrality.values()))
        # this should never be zero?
        except ZeroDivisionError:
            s = 1.0
        for n in Vector_Centrality:
            Vector_Centrality[n] *= s

        # check convergence
        err = sum([abs(Vector_Centrality[n] - Vector_Centrality_Old[n]) for n in Vector_Centrality])
        #print "Iteration:",i, sorted(Vector_Centrality_Old.iteritems(), key=lambda d:d[1], reverse = True)
        if err < nnodes*tol:
            return Vector_Centrality
    #end for
    print "Weighted_Hybrid_Diffusion_Centrality Iteration Error", err
    print sorted(Vector_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    print sorted(Vector_Centrality_Old.iteritems(), key=lambda d:d[1], reverse = True)





#Calculate average values of eigenvector based centralities based on multiple experiments
#用于在特定已知网络上对提出的中心性度量进行实验分析和性能对比 （多次执行累加后看总体表现）
def PIR_Centrality_Avg(G):
    PIR_Centrality = dict([(n, 0.0) for n in G])
    for i in range(10):
        Eigen_Ranking = Weighted_Hybrid_Diffusion_Centrality(G)
        for n in G:
            PIR_Centrality[n] = PIR_Centrality[n] + Eigen_Ranking[n]
        #end for
    #end for
    #print "PIR_Centrality",  sorted(PIR_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return PIR_Centrality



###*****************************************************************************
def Nodes_Ranking(G, index):
    if index == "degree_centrality":
        return Degree_Centrality(G)
    if index == "between_centrality":
        return Between_Centrality(G)
    if index == "closeness_centrality":
        return Closeness_Centrality(G)
    if index == "pagerank_centrality":
        return Page_Rank(G)
    if index == "kshell_centrality":
        return KShell_Centrality(G)
    if index == "collective_influence":
        return Collective_Influence(G)
    if index == "enhanced_collective_centrality":
        return Enhanced_Collective_Influence(G)

    if index == "eigen_centrality":
        return Eigen_Centrality_Avg(G) #Eigen_Centrality_Andy(G)

    if index == "md_eigen_centrality":
        return MD_Eigen_Centrality_Andy(G)
    if index == "hc_eigen_centrality":
        return HC_Eigen_Centrality_Andy(G)

    #if index == "hybrid_diffusion_centrality":
    #    return Hybrid_Diffusion_Centrality(G)


    if index == "PIR_Centrality": #i.e. weighted_hybrid_diffusion_centrality
        return PIR_Centrality_Avg(G) #Weighted_Hybrid_Diffusion_Centrality(G)