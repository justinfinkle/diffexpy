---
layout: index
---

# pydiffexp
Dynamic Differential Expression in Python

Python is great for data analysis, but lacks some of the bioninformatics tools available in R ([Bioconductor](https://bioconductor.org/)). `pydiffexp` not only makes differential expression analysis in Python easy, but it also allows users to conduct dynamic differential expresison analysis on time course data. 

`pydiffexp` uses [rpy2](http://rpy2.bitbucket.org/) to run limma analyis. All input and output is with python and [pandas](http://pandas.pydata.org/), while the differential expresison analysis object takes care of the model fitting in R.
