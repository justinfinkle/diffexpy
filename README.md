# pydiffexp
Differential Expression in Python

Python is great for data analysis, but lacks some of the bioninformatics tools available in R ([Bioconductor](https://bioconductor.org/)). pydiffexp is a simple package that uses [rpy2](http://rpy2.bitbucket.org/) to run limma analyis. All input and output is with python and [pandas](http://pandas.pydata.org/), while the differential expresison analysis object takes care of the model fitting in R.
