__author__ = 'taowu'
#!/user/bin/env python
#coding:utf-8
# -*- coding: utf-8 -*-

import linecache
import string
import os
import math
import time
import random
import networkx as nx
import  numpy as np
import matplotlib.pyplot as plt

from math import log
from numpy import array
import Louvain_community as lw

import Link_Predictors as Link_Predictors
import Data_Prepare as Data_Prepare
import Position_Drift as Position_Drift





def matploit(data):
    plt.figure(figsize=(8,5), dpi=80)
    plt.subplot(1,1,1)
    plt.grid()
    plt.subplots_adjust(top=0.9)

    X = [i for i in range(len(data))]

    plt.scatter(X,data,color="r")

    plt.xlim(0.0,len(data))
    #plt.xticks(np.linspace(1.0,4.0,7,endpoint=True))
    plt.ylim(0,max(data)+1)
    #plt.yticks(np.linspace(0.0,18,7,endpoint=True))

    plt.xlabel("Link number")
    plt.ylabel("Similarity")
    plt.show()



def Create_Graph(fname = None):
    '''
    G = nx.Graph()
    G.add_edge(0, 1, weight = 2.0,timestamp = 1.0)
    G.add_edge(0, 2, weight = 2.0,timestamp = 1.0)
    G.add_edge(0, 3, weight = 2.0,timestamp = 1.0)
    G.add_edge(0, 4, weight = 2.0,timestamp = 1.0)
    G.add_edge(0, 5, weight = 2.0,timestamp = 1.0)
    G.add_edge(4, 6, weight = 2.0,timestamp = 1.0)
    G.add_edge(4, 7, weight = 2.0,timestamp = 1.0)
    G.add_edge(4, 8, weight = 2.0,timestamp = 1.0)
    G.add_edge(7, 8, weight = 2.0,timestamp = 1.0)
    G.add_edge(5, 9, weight = 2.0,timestamp = 1.0)
    G.add_edge(2, 3, weight = 2.0,timestamp = 1.0)
    G.add_edge(2, 13, weight = 2.0,timestamp = 1.0)
    G.add_edge(2, 11, weight = 2.0,timestamp = 1.0)
    G.add_edge(2, 12, weight = 2.0,timestamp = 1.0)
    G.add_edge(11, 12, weight = 2.0,timestamp = 1.0)
    G.add_edge(3, 11, weight = 2.0,timestamp = 1.0)
    G.add_edge(3, 10, weight = 2.0,timestamp = 1.0)
    '''

    #fname = 'F:/Link_Prediction_Code/Dataset/6-Wireless_contact_Train_Regular.txt'
    #Get edge from txt type data
    try:
        fdobj = open(fname,'r')
    except IOError as e:
        print "***file open error:",e
    else:
        G = nx.Graph()
        eline = fdobj.readline()
        eline = fdobj.readline()
        eline = fdobj.readline()
        eline = fdobj.readline()
        eline = fdobj.readline()
        eline = fdobj.readline()
        eline = fdobj.readline()
        eline = fdobj.readline()
        eline = fdobj.readline()
        while eline:
            line = eline.strip('\n').split()
            G.add_edge(string.atoi(line[0]),string.atoi(line[1]),weight=string.atof(line[3]),timestamp=string.atof(line[4]))#weight=string.atof(line[3])
            eline = fdobj.readline()
        #end while

    #Data_Prepare.Drawcomgraph(G)
    return G








##==========================================================================================
'''
def Probe_Set_Correspond_Training(G, Top_L, fpname):
    #Considering the same node set with the training graph and the same number with the ranking list.
    print "Probe_Set_Correspond_Training!"
    try:
        fdobj = open(fpname,'r')
    except IOError as e:
        print "***file open error:",e
    else:
        Probe_Set = []

        ############
        FG = nx.Graph()
        eline = fdobj.readline()
        while eline:
            line = eline.strip('\n').split()
            if FG.has_edge(string.atoi(line[0]),string.atoi(line[1])):
                #Weight
                FG[string.atoi(line[0])][string.atoi(line[1])]['weight'] = FG[string.atoi(line[0])][string.atoi(line[1])]['weight'] + string.atof(line[2])
                #Time
                if FG[string.atoi(line[0])][string.atoi(line[1])]['timestamp'] > string.atof(line[3]):
                    FG[string.atoi(line[0])][string.atoi(line[1])]['timestamp'] = string.atof(line[3])
            else:
                FG.add_edge(string.atoi(line[0]),string.atoi(line[1]),weight=string.atof(line[2]),timestamp=string.atof(line[3]))
            eline = fdobj.readline()
        #end while

        List_Set = FG.edges(data='True')
        List_Sorted = sorted(List_Set, key=lambda edge: edge[2]['timestamp'])
        #print Top_L, len(List_Sorted), List_Sorted

        print List_Sorted[0],List_Sorted[1],List_Sorted[2]

        for i in range(0, len(List_Sorted)):
            if G.has_node(List_Sorted[i][0]) and G.has_node(List_Sorted[i][1]):
                if G.has_edge(List_Sorted[i][0], List_Sorted[i][1]):
                    print "ERROR: overlapping links in probe_set"
                else:
                    Probe_Set.append((List_Sorted[i][0],List_Sorted[i][1]))
        #end for

        if Top_L < len(Probe_Set):
            return Probe_Set[0:Top_L]
        else:
            print "ERROR: the test data is not enough (less than |taining|/20 )"
'''

##==========================================================================================
def Performance_Evaluation_AUC(Predictor, G, Probe_Set, Non_existing_links):
    Count = 0.0
    for i in range(0,len(Probe_Set)):
        future_link_score = Link_Predictors.Wighted_Link_Prediction(Predictor, G, random.choice(Probe_Set))[0][2]
        never_link_score = Link_Predictors.Wighted_Link_Prediction(Predictor, G, random.choice(Non_existing_links))[0][2]
        if future_link_score > never_link_score:
            Count = Count + 1.0
        elif future_link_score == never_link_score:
            Count = Count + 0.5
    #end for
    return Count/float(len(Probe_Set))

'''
def Performance_Evaluation_AUC(Predictor, G, Probe_Set):
    Unobserved_links = nx.non_edges(G)
    Non_existing_links = list(set(Unobserved_links).difference(set(Probe_Set)))

    Count = 0.0
    for i in range(0,len(Probe_Set)):
        future_link_score = Link_Predictors.Wighted_Link_Prediction(Predictor, G, random.choice(Probe_Set))[0][2]
        never_link_score = Link_Predictors.Wighted_Link_Prediction(Predictor, G, random.choice(Non_existing_links))[0][2]
        if future_link_score > never_link_score:
            Count = Count + 1.0
        elif future_link_score == never_link_score:
            Count = Count + 0.5
    #end for
    return Count/float(len(Probe_Set))
'''

def Performance_Evaluation_Precision(Top_L_Rank_List, Probe_Set): #[(u, v, s)], [(u,v)]
    #print len(Top_L_Rank_List)
    #print len(Probe_Set)

    Top_Link = []
    Future_Link = []
    if len(Top_L_Rank_List) == len(Probe_Set):
        for Link in Top_L_Rank_List:
            Top_Link.append((Link[0], Link[1]))
        for Link in Probe_Set:
            Future_Link.append((Link[0], Link[1]))
        #print Top_Link
        #print Top_Link
        #print Future_Link
        Common_links = list(set(Top_Link).intersection(set(Future_Link)))
        return float(len(Common_links))/float(len(Future_Link))


##==========================================================================================

#Prediction with different training set proportion
def Prediction_LinkScores_Ratio(G, Predictor, Proportion, Toleration, Predict_Gap):
    print "Prediction_LinkScores_Ratio!"
    Rank_List_Set = {}
    OK_Value = float(G.number_of_edges())/Proportion

    if nx.is_connected(G) == True:
        Edge_Set = G.edges(data='True')

        Total = 0
        Error = 0
        Rank_List_Set[0] = [Link_Predictors.Wighted_Link_Prediction(Predictor, G), nx.average_clustering(G), nx.average_shortest_path_length(G) ]  ##Running time !!!!!
        '''
        while 1:
            #print i,len(Edge_Set),
            Tep_Edge = []
            Del = random.randint(0, len(Edge_Set)-1)
            Tep_Edge.append(Edge_Set[Del])

            #print "random range:", len(Edge_Set)-1
            #print Del,

            #Prediction with different training set
            G.remove_edge(Edge_Set[Del][0], Edge_Set[Del][1])
            if nx.is_connected(G) != True:
                G.add_edges_from(Tep_Edge)
                Error = Error + 1
                #print "Error:", Error
            else:
                #print Edge_Set[Del],
                Error = 0
                Total = Total + 1
                #print "Total:", Total

                if Total%Predict_Gap == 0:
                    V1 = Link_Predictors.Wighted_Link_Prediction(Predictor, G)
                    V2 = nx.average_clustering(G)
                    V3 = nx.average_shortest_path_length(G)
                    #V4 = Performance_Evaluation_AUC(Predictor, G, Probe_Set, Non_existing_links)
                    Rank_List_Set[Total] = [V1,V2,V3]
                Edge_Set = G.edges(data='True')
            #end if
            if Total > OK_Value or Error == Toleration:
                #print "complete with Total, Error:", Total, Error
                return Rank_List_Set
        #end while
        '''
        return Rank_List_Set
    #end if
    #return Rank_List_Set

##==========================================================================================
#Native_Prediction_Experiment(G, 'WSD', Probe_Set, Top_L, 3) #Top_K, Deleted_Ratio
def Prediction_Experiment(G, Predictor, Probe_Set, Top_L, Deleted_Ratio):
    print "Prediction_Experiment!"
    #Get Evaluation Link Set--------
    #Top_L = (G.number_of_edges() - 0) / Top_k #The top proportion 1/Top_k of edges are considered
    #Probe_Set = Probe_Set_Correspond_Training(G, Top_L, fpname)  #****Get the probe set for evaluation*****
    #Get Ranking List with different deleted links ratio----------
    Edge_Num = float(G.number_of_edges())

    '''AUC = Performance_Evaluation_AUC(Predictor, G, Probe_Set)'''
    Unobserved_links = nx.non_edges(G)
    Non_existing_links = list(set(Unobserved_links).difference(set(Probe_Set)))
    AUC = Performance_Evaluation_AUC(Predictor, G, Probe_Set, Non_existing_links)

    Rank_List_Set = Prediction_LinkScores_Ratio(G, Predictor, Deleted_Ratio, 50, 30) #Prediction_LinkScores_Ratio(G, Predictor, Proportion, Toleration, Predict_Gap)
    #----Performance Evaluation with Precision under different Training Data Ratio----
    Precision_Set = []
    X_Set = []
    Coefficient_Set = []
    Avg_PathLen_Set = []
    for key in sorted(Rank_List_Set.keys()):
        Rank_List_Sorted = sorted(Rank_List_Set[key][0], key=lambda edge: edge[2], reverse=True)
        Top_L_Rank_List = Rank_List_Sorted[0:Top_L]
        Coefficient_Set.append(Rank_List_Set[key][1])
        Avg_PathLen_Set.append(Rank_List_Set[key][2])
        #AUC_Set.append(Rank_List_Set[key][3])
        #print key, Performance_Evaluation_Precision(Top_L_Rank_List, Probe_Set)
        X_Set.append(float(key)/Edge_Num)
        Precision_Set.append(Performance_Evaluation_Precision(Top_L_Rank_List, Probe_Set))
        '''
        #Draw Curve Graph
        if key%100 == 0:
            data = []
            for edge in Rank_List_Sorted:
                data.append(edge[2])
            matploit(data)
        '''
    #end for
    print "*Different deleted links ratio:", X_Set
    print "*Precision_Set with different deleted links ratio:", Precision_Set
    print "*Coefficient_Set:", Coefficient_Set
    print "*Avg_PathLen_Set:", Avg_PathLen_Set
    print "*AUC Value:", AUC


    return 1




#def Native_Prediction_Experiment(G, Predictor, Probe_Set, Top_L, Deleted_Ratio):
def Drift_Prediction_Experiment(G, Predictor, Probe_Set, Top_L, Deleted_Ratio):
    print "Drift_Prediction_Experiment!"
    Edge_Num = float(G.number_of_edges())

    #AUC = Performance_Evaluation_AUC(Predictor, G, Probe_Set)
    Unobserved_links = nx.non_edges(G)
    #Unobserved_links = list(Unobserved_links)
    #print Unobserved_links
    #print Probe_Set


    Non_existing_links = list(set(Unobserved_links).difference(set(Probe_Set)))
    AUC = Performance_Evaluation_AUC(Predictor, G, Probe_Set, Non_existing_links)

    #***Prediction with different training set proportion***
    t1 = time.time()
    Rank_List_Set = Prediction_LinkScores_Ratio(G, Predictor, Deleted_Ratio, 50, 30) #Prediction_LinkScores_Ratio(G, Predictor, Proportion, Toleration, Predict_Gap)
    t2 = time.time()
    print "Prediction index time",t2-t1

    #----Performance Evaluation with Precision under different Training Data Ratio----
    Precision_Set = []
    X_Set = []
    Coefficient_Set = []
    Avg_PathLen_Set = []
    for key in sorted(Rank_List_Set.keys()):
        Rank_List_Sorted = sorted(Rank_List_Set[key][0], key=lambda edge: edge[2], reverse=True)
        Top_L_Rank_List = Rank_List_Sorted[0:Top_L]
        Coefficient_Set.append(Rank_List_Set[key][1])
        Avg_PathLen_Set.append(Rank_List_Set[key][2])
        X_Set.append(float(key)/Edge_Num)
        Precision_Set.append(Performance_Evaluation_Precision(Top_L_Rank_List, Probe_Set))
    #end for
    print "*Drift_Different deleted links ratio:", X_Set
    print "*Drift_Precision_Set with different deleted links ratio:", Precision_Set
    print "*Drift_Coefficient_Set:", Coefficient_Set
    print "*Drift_Avg_PathLen_Set:", Avg_PathLen_Set
    print "*Drift_AUC Value:", AUC

    return 1









#=========================================================================================

def Connected_Prediction_Main(Predictor, fnid):
    try:
        fnobj = open(fnid,'r')
    except IOError as e:
        print "***file open error:",e
    else:
        #**************Read data and Create graph*******************
        G = nx.Graph()
        eline = fnobj.readline()
        eline = fnobj.readline()
        while eline:
            line = eline.strip().split()
            #print line
            #edge = (line[0],line[1],line[2],line[3])
            #Edge.append(tep)
            if G.has_edge(string.atoi(line[0]),string.atoi(line[1])):
                G[string.atoi(line[0])][string.atoi(line[1])]['weight'] = G[string.atoi(line[0])][string.atoi(line[1])]['weight'] + string.atof(line[2])
                if G[string.atoi(line[0])][string.atoi(line[1])]['timestamp'] < string.atof(line[3]):
                    G[string.atoi(line[0])][string.atoi(line[1])]['timestamp'] = string.atof(line[3])
            else:
                G.add_edge(string.atoi(line[0]),string.atoi(line[1]),weight=string.atof(line[2]),timestamp=string.atof(line[3]))
            eline = fnobj.readline()
        #end while

        ################################################
        partition = lw.best_partition(G)
        print partition.keys()

        DrawGraph = {}
        for nid in partition.keys():
            if DrawGraph.has_key(partition[nid]):
                DrawGraph[partition[nid]].append(nid)
            else:
                DrawGraph[partition[nid]] = [nid]
        #enf for
        print len(DrawGraph)

        for com in DrawGraph.keys():
            print com,len(DrawGraph[com])
        #print DrawGraph[5]

        #-------------------------------------------------
        for com in DrawGraph.keys():
            if com != 17:
                for node in DrawGraph[com]:
                    G.remove_node(node)
        #end for
        print G.number_of_nodes()
        ###################################################


        #********Split data into training data and test data*******
        Edges_Set = G.edges(data=True)
        #print Edges_Set
        Sorted_Edges_Set = sorted(Edges_Set, key=lambda edge: edge[2]['timestamp'])
        #print Sorted_Edges_Set

        Top_L = int(len(Sorted_Edges_Set)*0.08)
        #Training_Set = Sorted_Edges_Set[:-(Top_L+1)]
        Test_Set = Sorted_Edges_Set[-Top_L:]
        #print Test_Set
        #-------------------------------------------

        #*****Ensuring the greatest component of training network structure******
        for edge in Test_Set:
            G.remove_edge(edge[0],edge[1])
        if nx.is_connected(G) == False:
            ### Get the Greatest Component of Networks #####
            components = sorted(nx.connected_components(G), key = len, reverse=True)
            print len(components)
            for i in range(1, len(components)):
                for node in components[i]:
                    G.remove_node(node)
            #end for
            print nx.is_connected(G)
            print G.number_of_nodes()
        #end if------------------------------



        #*****Ensuring the training data and test data have the same nodes set******
        Probe_Set = []
        for i in range(0, len(Test_Set)):
            if G.has_node(Test_Set[i][0]) and G.has_node(Test_Set[i][1]):
                if G.has_edge(Test_Set[i][0], Test_Set[i][1]):
                    print "ERROR: overlapping links in training data and test data"
                else:
                    Probe_Set.append((Test_Set[i][0],Test_Set[i][1], Test_Set[i][2]['timestamp']))
        #end for
        Top_L = len(Probe_Set)
        print "len(Probe_Set):",len(Probe_Set)
        print "len(Test_Set):",len(Test_Set)
        print "len(G.edges()):",len(G.edges())
        print "len(G.nodes()):",len(G.nodes())
        #print Test_Set

        #####*********Statistic attributes of graphs*************
        print "*Statistic attributes of graphs:"
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

        #Regularization
        #for edge in G.edges(data=True):
        #    edge[2]['weight'] = 1#math.exp( -1 / edge[2]['weight'] ) + math.exp( -1 / edge[2]['timestamp'] )
        #end for
        #*******************************************************************************#
        print fnid
        #print "============Native_Prediction_Experiment==============================="
        #Drift_Prediction_Experiment(G, Predictor, Probe_Set, Top_L, 3.0)
        #Prediction_Experiment(G, Predictor, Probe_Set, Top_L, 3.0) #Top_K, Deleted_Ratio

        #print "============Drift_Prediction_Experiment==============================="
        #G = Position_Drift.Graph_Spatial_Temporal_Dynamics(G, 3)  #Spatial_Temporal influence based node position drift.
        #Prediction_Experiment(G, Predictor, Probe_Set, Top_L, 3.0) #Top_K, Deleted_Ratio
        Drift_Prediction_Experiment(G, Predictor, Probe_Set, Top_L, 2.5)







#=======================================================================================
def Prediction_Main(Predictor, fnid):
    try:
        fnobj = open(fnid,'r')
    except IOError as e:
        print "***file open error:",e
    else:
        #**************Read data and Create graph*******************
        G = nx.Graph()

        eline = fnobj.readline()
        eline = fnobj.readline()
        while eline:
            line = eline.strip().split()
            #print line
            #edge = (line[0],line[1],line[2],line[3])
            #Edge.append(tep)
            if G.has_edge(string.atoi(line[0]),string.atoi(line[1])):
                G[string.atoi(line[0])][string.atoi(line[1])]['weight'] = G[string.atoi(line[0])][string.atoi(line[1])]['weight'] +   (string.atof(line[2]) + 2)
                if G[string.atoi(line[0])][string.atoi(line[1])]['timestamp'] < string.atof(line[3]):
                    G[string.atoi(line[0])][string.atoi(line[1])]['timestamp'] = string.atof(line[3])
            else:
                G.add_edge(string.atoi(line[0]), string.atoi(line[1]), weight=(string.atof(line[2])+2), timestamp=string.atof(line[3]))
            eline = fnobj.readline()
        #end while

        #********Split data into training data and test data*******
        Edges_Set = G.edges(data=True)
        Sorted_Edges_Set = sorted(Edges_Set, key=lambda edge: edge[2]['timestamp'])

        Top_L = int(len(Sorted_Edges_Set)*0.08)
        #Training_Set = Sorted_Edges_Set[:-(Top_L+1)]
        Test_Set = Sorted_Edges_Set[-Top_L:]

        #*****Ensuring the greatest component of training network structure******
        #print nx.is_connected(G)         #print len(Sorted_Edges_Set)        print len(G.nodes())
        for edge in Test_Set:
            G.remove_edge(edge[0],edge[1])
        if nx.is_connected(G) == False:
            ### Get the Greatest Component of Networks #####
            components = sorted(nx.connected_components(G), key = len, reverse=True)
            #print len(components)
            for i in range(1, len(components)):
                for node in components[i]:
                    G.remove_node(node)
            #end for
            print nx.is_connected(G)
        #end if

        #*****Ensuring the training data and test data have the same nodes set******
        Probe_Set = []
        for i in range(0, len(Test_Set)):
            if G.has_node(Test_Set[i][0]) and G.has_node(Test_Set[i][1]):
                if G.has_edge(Test_Set[i][0], Test_Set[i][1]):
                    print "ERROR: overlapping links in training data and test data"
                else:
                    Probe_Set.append((Test_Set[i][0],Test_Set[i][1], Test_Set[i][2]['timestamp']))
        #end for
        Top_L = len(Probe_Set)

        '''
        print "len(Probe_Set):",len(Probe_Set)
        print "len(Test_Set):",len(Test_Set)
        print "len(G.edges()):",len(G.edges())
        print "len(G.nodes()):",len(G.nodes())
        #print Test_Set

        #####*********Statistic attributes of graphs*************
        print "*Statistic attributes of graphs:"
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
        '''
        #Regularization
        #for edge in G.edges(data=True):
        #    edge[2]['weight'] = 1#math.exp( -1 / edge[2]['weight'] ) + math.exp( -1 / edge[2]['timestamp'] )
        #end for
        #*******************************************************************************#
        print fnid
        #print "============Native_Prediction_Experiment==============================="
        #Drift_Prediction_Experiment(G, Predictor, Probe_Set, Top_L, 3.0)
        #Prediction_Experiment(G, Predictor, Probe_Set, Top_L, 3.0) #Top_K, Deleted_Ratio

        #print "============Drift_Prediction_Experiment==============================="
        #t1 = time.time()
        #G = Position_Drift.Graph_Spatial_Temporal_Dynamics(G,6)  #Spatial_Temporal influence based node position drift.
        #t2 = time.time()
        #print "Drift time:", t2 - t1
        #Prediction_Experiment(G, Predictor, Probe_Set, Top_L, 3.0) #Top_K, Deleted_Ratio
        Drift_Prediction_Experiment(G, Predictor, Probe_Set, Top_L, G.number_of_edges())
        #G.number_of_edges()
    fnobj.close()





##==========================================================================================
if __name__ == '__main__':
    fnid = 'F:/Link_Prediction_Code/Dataset/5-Infectious.txt'#
    Prediction_Main('WSD', fnid)
    #Connected_Prediction_Main('WAA', fnid)

    '''
    Predictors = ['WCN','WAA','WRA', 'WLP', 'WSD']#G, 'WAA', Probe_Set, Top_L, 4
    for Predictor in Predictors:
        #Native_Prediction_Experiment(G, Predictor, Probe_Set, Top_L, 2)
        Prediction_Main(Predictor, fnid)
        print
        print "==========================================="
    #end for
    Predictors = ['WCN','WAA','WRA', 'WLP', 'WSD']#G, 'WAA', Probe_Set, Top_L, 4
    for Predictor in Predictors:
        Drift_Prediction_Experiment(G, Predictor, Probe_Set, Top_L, 2)
        print
        print "==========================================="
    #end for
    '''