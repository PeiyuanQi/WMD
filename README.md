# WMD Wrapper Implementation

This is an implementation of Wrapper for WMD (Word Move Distance) based on gensim lib in python.

## Requirement
- Python3 and supporting packages
- Trained doc2vec or word2vec model

## Usage
Run `wmd.py` with following args:

- Docs directories `-l <testdocs dir>`;
- Model location `-m <model dir>`;
- Aggregation threshold `-t <threshold>`;
- Output file `-o <output.csv>`.

Output:
Clustered results in specified output file.

## Reference
- Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, and Kilian Q. Weinberger. 2015.
[From word embeddings to document distances.](https://arxiv.org/pdf/1409.3215.pdf)

    To cite using the following BibTeX entry (instead of Google Scholar): 
    
        @inproceedings{kusner2015doc, 
           title={From Word Embeddings To Document Distances}, 
           author={Kusner, M. J. and Sun, Y. and Kolkin, N. I. and Weinberger, K. Q.}, 
           booktitle={ICML}, 
           year={2015}, 
        } 
- [gensim](https://radimrehurek.com/gensim/models/keyedvectors.html).