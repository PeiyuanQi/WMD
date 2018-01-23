#!/home/utils/Python-3.6.1/bin/python3
"""
A warpped python script to easily aggregate the vectors
"""

import os
import sys
import numpy as np
import pandas as pd
import optparse
from gensim.models import Doc2Vec
from nltk.tokenize import word_tokenize

def GetOptions():
    parser = optparse.OptionParser()
    parser.add_option('-l', '--list',
                      dest="logdir",
                      help="input docs dir.",
                      )
    parser.add_option('-m', '--model',
                      dest="model",
                      help="saved model.",
                      )
    parser.add_option('-t', '--threshold',
                      dest="threshold",
                      help="threshold value.",
                      )
    parser.add_option('-o', '--outfile',
                      dest="outdir",
                      help="output csv dir.",
                      )
    options, remainder = parser.parse_args()
    return options

options = GetOptions()


def get_log_list(logs_dir):
    """Get all the log file name."""
    log_list = []
    for fname in os.listdir(logs_dir):
        log_list.append(fname)
    return log_list

def start_clustering(distance_matrix, threshold, logs_num):
    cluster_id = [-1] * logs_num
    cur_id = 0
    same_cluster_id_pair = []
    for i in range(0, logs_num):
        id_for_j = cur_id
        if (cluster_id[i] < 0):
            cluster_id[i] = cur_id
            cur_id += 1
        else:
            id_for_j = cluster_id[i]
        for j in range(i+1, logs_num):
            if (distance_matrix[i,j] <= threshold):
                if (cluster_id[j] < 0):
                    cluster_id[j] = id_for_j
                else:
                    # cur_id should be > cluster_id[i]
                    if (cur_id != cluster_id[i]):
                        aTuple = (cluster_id[i],cur_id)
                        # there should be no duplicate tuple
                        same_cluster_id_pair.append(aTuple)
    for i in range(0, logs_num):
        for j in range(0, len(same_cluster_id_pair)):
            if (cluster_id[i] == same_cluster_id_pair[j][1]):
                cluster_id[i] = same_cluster_id_pair[j][0]
    return cluster_id


def add_error_log_column(sourceFileName,outputFileName):
    baseLogDir = options.logdir
    df = pd.read_csv(sourceFileName)
    df = df.assign(errorLog=' ')
    for index, row in df.iterrows():
        logDir = baseLogDir + row['DocTag']
        # print(row['DocTag'])
        with open(logDir,"r") as fi:
            df.loc[index, "errorLog"] = fi.read().replace("\n"," ").replace(","," ").replace("\r"," ").replace("\t"," ")
            # print(row['errorLog'])
            fi.close()
    df.to_csv(outputFileName, sep=',')

# ploting codes are commented out for not able to run on engineering space on cloud.
# import matplotlib.pyplot as plt
# import networkx as nx
#
#
# def plot_weighted_graph(log_list,distance, logs_num):
#     # refer: https://qxf2.com/blog/drawing-weighted-graphs-with-networkx/
#     G = nx.Graph()  # Create a graph object called G
#     node_list = log_list
#     for node in node_list:
#         G.add_node(node)
#
#     # Note: You can also try a spring_layout
#     pos = nx.circular_layout(G)
#     nx.draw_networkx_nodes(G, pos, node_color='green', node_size=7500)
#
#     # 3. If you want, add labels to the nodes
#     labels = {}
#     for node_name in node_list:
#         labels[str(node_name)] = str(node_name)
#     nx.draw_networkx_labels(G, pos, labels, font_size=8)
#
#     # 4. Add the edges (4C2 = 6 combinations)
#     for i in range(0, logs_num):
#         for j in range(i+1,logs_num):
#             G.add_edge(node_list[i], node_list[j], weight=distance[i,j])  # Karpov vs Kasparov
#
#     all_weights = []
#     # 4 a. Iterate through the graph nodes to gather all the weights
#     for (node1, node2, data) in G.edges(data=True):
#         all_weights.append(data['weight'])  # we'll use this when determining edge thickness
#
#     # 4 b. Get unique weights
#     unique_weights = list(set(all_weights))
#
#     # 4 c. Plot the edges - one by one!
#     for weight in unique_weights:
#         # 4 d. Form a filtered list with just the weight you want to draw
#         weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in G.edges(data=True) if
#                           edge_attr['weight'] == weight]
#         # 4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
#         width = weight * len(node_list) * 3.0 / sum(all_weights)
#         nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, width=width)
#
#     # Plot the graph
#     plt.axis('off')
#     plt.title('Distance Cluster')
#     plt.savefig("wmd.png")



if __name__ == '__main__':
    logs_dir = options.logdir
    model_dir = options.model
    threshold = int(options.threshold)
    output_dir = options.outdir

    model = Doc2Vec.load(model_dir)
    log_list = get_log_list(logs_dir)

    # generate graph and index id, edge and so on.
    print("generating wmd matrix...")
    logs_num = len(log_list)
    distance = np.zeros((logs_num, logs_num))
    for i in range(0, logs_num):
        tok_list_1 = []
        with open(logs_dir + os.sep + log_list[i], "r") as fi:
            sent = fi.read()
            tok_list_1 = word_tokenize(sent)
            print(log_list[i] + " read")
            fi.close()
        for j in range(i+1, logs_num):
            tok_list_2 = []
            with open(logs_dir + os.sep + log_list[j],"r") as fi:
                sent = fi.read()
                tok_list_2 = word_tokenize(sent)
                print(log_list[j] + " read")
                fi.close()
            distance[i,j] = model.wmdistance(tok_list_1,tok_list_2)
            print("dis between i, j: " + str(distance[i,j]))
    np.save('tmp.txt',distance)

    print("start clustering...")
    array_of_cluster_id = start_clustering(distance, threshold, logs_num)

    print("formating outputs...")
    result = pd.DataFrame({'DocTag': log_list, 'ClusterId': array_of_cluster_id}, columns=['DocTag', 'ClusterId'])
    result.to_csv("medium.csv", index=False)

    print("saving results...")
    add_error_log_column("medium.csv", output_dir)
    # plot_weighted_graph(log_list, distance,logs_num)

