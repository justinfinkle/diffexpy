# DiffExPy

### Please Note: This repository is currently in the process of being refactored. Branches will be merged into `master` when refactoring is completed

Dynamic Differential Expression in Python

Python is great for data analysis, but lacks some of the bioninformatics tools available in R ([Bioconductor](https://bioconductor.org/)). `PyDiffExp` not only makes differential expression analysis in Python easy, but it also allows users to conduct dynamic differential expresison analysis on time course data. `diffexpy` uses [rpy2](http://rpy2.bitbucket.org/) to run limma analyis. All input and output is with python and [pandas](http://pandas.pydata.org/), while the differential expresison analysis object takes care of the model fitting in R.
