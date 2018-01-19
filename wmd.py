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
from dijkstra import *

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


if __name__ == '__main__':
    options = GetOptions()
    logs_dir = options.logdir
    model_dir = options.model
    threshold = options.threshold
    output_dir = options.outdir

    model = Doc2Vec.load(model_dir)
    unclusted_log_list = get_log_list(logs_dir)
    cluster_id = 0

    while (len(unclusted_log_list)>0):
        cur_doc = unclusted_log_list.pop()
        sent = word_tokenize(cur_doc)