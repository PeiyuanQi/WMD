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


def get_log_list(logs_dir):
    """Get all the log file name."""
    log_list = []
    for fname in os.listdir(logs_dir):
        log_list.append(fname)
    return log_list

def start_clustering(distance_matrix, threshold, log_list, logs_num):
    cluster_id = [-1] * logs_num
    cur_id = 0
    same_cluster_id_pair = []
    for i in range(0, logs_num):
        id_for_j = cur_id
        if (cluster_id[i] < 0):
            cluster_id[i] = cur_id
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
    baseLogDir = './DataSet/test/filteredlogs/'
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


if __name__ == '__main__':
    options = GetOptions()
    logs_dir = options.logdir
    model_dir = options.model
    threshold = options.threshold
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
    np.save('tmp.txt',distance)

    print("start clustering...")
    array_of_cluster_id = start_clustering(distance, threshold, log_list, logs_num)

    print("formating outputs...")
    result = pd.DataFrame({'DocTag': log_list, 'ClusterId': array_of_cluster_id}, columns=['DocTag', 'ClusterId'])
    result.to_csv("medium.csv", index=False)

    print("saving results...")
    add_error_log_column("medium.csv", options.output_dir)

