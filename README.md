# WMD Wrapper Implementation

This is an implementation of Wrapper for WMD (Word Move Distance) Algorithm based on gensim lib in python.

## Requirement
- Python3 and supporting packages
- Trained doc2vec or word2vec model

## Usage
Run `wmd.py` with following args:

docs directories `-l <dir>`, Aggregation threshold `-t <threshold>`. Output file `-o <output.csv>`.

Output:
Clustered results in specified output file.

## Reference
- Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, and Kilian Q. Weinberger. 2015.
[From word embeddings to document distances.](https://arxiv.org/pdf/1409.3215.pdf)
In Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37 (ICML'15), 
Francis Bach and David Blei (Eds.), Vol. 37. JMLR.org 957-966.
- [gensim](https://radimrehurek.com/gensim/models/doc2vec.html)