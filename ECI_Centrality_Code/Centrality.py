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

#****************************************************************************************
#IRA-迭代式资源分配方法for influential spreaders检测
def Iterative_Allocation():
    #目标：发现influential spreaders.
    Edge = [('0', '1'), ('0', '2'), ('1', '2'), ('0', '3'), ('0', '4'), ('2', '3')]
    G = nx.Graph()
    G.add_edges_from(Edge)

    KShell = {}
    KShell['0'] = 1
    KShell['1'] = 1
    KShell['2'] = 1
    KShell['3'] = 1
    KShell['4'] = 1
    Resource_Centrality = {}
    for nid in G.nodes():
        Resource_Centrality[nid] = 0.0
    #end for
    Resource_Centrality['0'] = 1.0

    def Total_KShell(ngb):
        Total_KShell = 0.0
        for nid in G.neighbors(ngb):
            Total_KShell = Total_KShell + KShell[nid]
        return Total_KShell

    #进行迭代式资源分配
    Flag = False
    while Flag == False:
        Flag = True
        for nid in G.nodes():
            neighbors = G.neighbors(nid)
            Updated_Resource_Nid = 0.0
            for ngb in neighbors:
                Updated_Resource_Nid = Updated_Resource_Nid + Resource_Centrality[ngb] * (KShell[nid]/Total_KShell(ngb))
            #end for
            if Resource_Centrality[nid] - Updated_Resource_Nid > 0.000001:
                Flag = False
            #Update
            Resource_Centrality[nid] = Updated_Resource_Nid
        #end for
    #end while
    return Resource_Centrality


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

#****************************************************************************************
def Local_Centrality(G): #局部节点平均度按比例加权和
    Local_Centrality_Dic = {}
    node_set = G.nodes()
    for nid in node_set:
        degree = G.degree(nid)
        if degree > 0:
            Neighbor_Set = G.neighbors(nid)
            total_neighbor_degree = 0
            for ngb in Neighbor_Set:
                total_neighbor_degree = total_neighbor_degree + G.degree(ngb)
            Local_Centrality_Dic[nid] = float(degree) + 0.5 * total_neighbor_degree / float(degree)
        else:
            Local_Centrality_Dic[nid] = 0
    #end for
    return Local_Centrality_Dic

###*****************************************************************************
def Degree_Mass_Centrality(G, m=2):
    #The 0th-order degree mass is the degree centrality, and the 
    #1th-order degree mass of node i is the sum of the degree of node i and the degree of its nearest neighbors.
    Degree_Mass_Centrality = {}
    node_set = G.nodes()
    for nid in node_set:#对于网路中每个节点。
        Degree_Mass_Centrality[nid] = 0
        NodeSet = G.neighbors(nid)
        NodeSet.append(nid)
        for ngb in G.neighbors(nid):
            NodeSet = list(set(NodeSet).union(set(G.neighbors(ngb))))
        #endfor
        #print "result:",nid, NodeSet
        HG = G.subgraph(NodeSet).copy() #子图
        for nd in HG.nodes():
            Degree_Mass_Centrality[nid] = Degree_Mass_Centrality[nid] + G.degree(nd)
        #end for
    #endfor
    return Degree_Mass_Centrality #{'1': 15, '0': 16, '3': 16, '2': 16, '5': 14, '4': 14, '6': 12}



#****************************************************************************************
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


#*****************************************************************************
def Mass_Diffusion_Centrality(G):
    #初始化
    Mass_Diffusion_Centrality = {}
    for nid in G.nodes():
        Mass_Diffusion_Centrality[nid] = 1.0
    #end for

    #进行迭代式资源分配
    def Iterative_Update_Mass_Diffusion(G, Resource_Centrality):
        Count = 0
        while Count > 0.95 * len(G.nodes()):
            for nid in G.nodes():
                neighbors = G.neighbors(nid)
                Updated_Resource_Nid = 0.0
                for ngb in neighbors:
                    Updated_Resource_Nid = Updated_Resource_Nid + Resource_Centrality[ngb] * (1.0/G.degree(ngb)) #邻居节点按比例分量之和！！
                #end for
                #Update
                if math.abs(Resource_Centrality[nid] - Updated_Resource_Nid) < 0.000001:
                    Count = Count + 1
                Resource_Centrality[nid] = Updated_Resource_Nid
            #end for
        #end while
        return Resource_Centrality

    #物质传播Mass Diffusion过程
    Mass_Diffusion_Centrality = Iterative_Update_Mass_Diffusion(G, Mass_Diffusion_Centrality)
    #print sorted(Mass_Diffusion_Centrality.iteritems(), key=lambda d:d[1], reverse = True)

    return Mass_Diffusion_Centrality




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


#**********************************************************************************



##******************************************************************************
def Enhanced_Collective_Influence_Native(G, d):
    #强化的Collective Influence, 参数d为考虑的范围radius。
    Enhanced_Collective_Influence_Dic = {}
    node_set = G.nodes()
    for nid in node_set: #对于网路中每个节点, 返回Center_set为距离中心节点 “d层”的节点集合。

        t1 = time.time()
        #循环过程的初始源和邻居
        Interior_Center_set = []
        Center_set = [nid] #关键层
        Neighbor_set = []
        #print nid, Center_set, d
        for i in range(0, d):
            for center in Center_set:
                Neighbor_set = list(set(Neighbor_set).union(G.neighbors(center)))
                #print "neighbors:", center, Neighbor_set, G.neighbors(center)
            #end for

            #更新源和邻居集合
            #从当前层的邻居节点集合中删除内层节点和本层节点，获得外层节点集合
            Temp_Set = list(set(list(set(Neighbor_set).difference(set(Interior_Center_set)))).difference(set(Center_set)))
            Interior_Center_set = Center_set
            Center_set = Temp_Set
            Neighbor_set = []
        #end for

        #(1)计算Collective_Influence取值
        Total_Reduced_Degree = 0.0
        for id in Center_set:
            Total_Reduced_Degree = Total_Reduced_Degree + (G.degree(id)-1.0)
        #end
        Collective_Influence = (G.degree(nid)-1.0) * Total_Reduced_Degree

        #(2)对nid的Collective_Influence进行关于Structure division(Hole spanners)的检测强化
        Associations_Ngb_Area_2 = 0
        for id1 in Center_set: #Center_set：离中心源点不同层的节点集合
            for id2 in Center_set:
                if id1 != id2:
                    NGB1 = list(set(G.neighbors(id1)).difference(set([nid])))
                    NGB1.append(id1)
                    NGB2 = list(set(G.neighbors(id2)).difference(set([nid])))
                    #NGB2.append(id2)
                    #print NGB2, NGB2, set(NGB1).intersection(set(NGB2))
                    Associations_Ngb_Area_2 = Associations_Ngb_Area_2 + len(set(NGB1).intersection(set(NGB2)))
                    #print "xijie", nid, Center_set, id1, id2, NGB1, NGB2, len(set(NGB1).intersection(set(NGB2)))
        #end for
        Associations_Ngb_Area_1 = 0
        if len(Interior_Center_set) > 1:
            for id1 in Interior_Center_set: #Center_set：离中心源点不同层的节点集合
                for id2 in Interior_Center_set:
                    if id1 != id2:
                        NGB1 = list(set(G.neighbors(id1)).difference(set([nid])))
                        NGB1.append(id1)
                        NGB2 = list(set(G.neighbors(id2)).difference(set([nid])))
                        Associations_Ngb_Area_1 = Associations_Ngb_Area_1 + len(set(NGB1).intersection(set(NGB2)))
        #end if

        #(3)节点nid的邻居结构的均衡性-structural entropy
        Degree_List = []
        Total_Degree = 0
        for node in G.neighbors(nid):
            Degree_List.append(G.degree(node))
            Total_Degree = Total_Degree + G.degree(node)
        #end for
        #print nid, Degree_List
        for i in range(0,len(Degree_List)):
            Degree_List[i] = Degree_List[i]/float(Total_Degree)
        #end for

        #计算熵
        Entropy = 0.0
        for i in range(0,len(Degree_List)):
            #print " Degree_List[i]:",  Degree_List[i]
            Entropy = Entropy + ( - Degree_List[i] * math.log( Degree_List[i] ) + 0.1 )
        #print G.degree(nid)
        Entropy = Entropy / math.log( G.degree(nid) + 0.1 )
        #end for

        #计算Enhanced_Collective_Influence
        '''是否对熵进行正则化处理？'''
        #Enhanced_Collective_Influence_Dic[nid] = [Collective_Influence / ((Associations_Ngb_Area_1+Associations_Ngb_Area_2)/2.0 + 0.001), Entropy]  # [Collective_Influence, Associations_Ngb_Area/2]
        Enhanced_Collective_Influence_Dic[nid] = Entropy * Collective_Influence / ((Associations_Ngb_Area_1 + Associations_Ngb_Area_2) / 2.0 + 0.001)  # [Collective_Influence, Associations_Ngb_Area/2]
        #print nid, Collective_Influence,Center_set,( Associations_Ngb_Area_1/2.0 + 0.001), ( Associations_Ngb_Area_2/2.0 + 0.001), Entropy

        t2=time.time()
        #print "nid:",nid,t2-t1
    #end for

    return Enhanced_Collective_Influence_Dic




###*****************************************************************************
def Nodes_Ranking(G, index):
    #Katz_Centrality = nx.katz_centrality(G)
    #print "Katz_Centrality:", sorted(Katz_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    #Page_Rank(G)
    if index == "degree_centrality":
        return Degree_Centrality(G)
    if index == "degree_mass_Centrality":
        return Degree_Mass_Centrality(G)
    if index == "between_centrality":
        return Between_Centrality(G)
    if index == "closeness_centrality":
        return Closeness_Centrality(G)
    if index == "kshell_centrality":
        return KShell_Centrality(G)
    if index == "eigen_centrality":
        return Eigen_Centrality_Andy(G)
    if index == "collective_influence":
        return Collective_Influence(G)
    if index == "enhanced_collective_centrality":
        return Enhanced_Collective_Influence(G)
    if index == "hybrid_diffusion_centrality":
        return Hybrid_Diffusion_Centrality(G)
