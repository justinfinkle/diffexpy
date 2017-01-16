__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import os
import sys
import pandas as pd

def make_path_iter(directory):
    pathiter = (os.path.join(root, filename) for root, _, filenames in os.walk(directory) for filename in
                filenames if root==directory)
    return pathiter

def assess_go_enrichment(directory, p_threshold):
    pathiter = make_path_iter(directory)
    go_enrich_dict = {}
    count = 0
    enrich_string = 'enrichment_'
    for path in pathiter:
        if 'whole' in path:
            continue
        if enrich_string in path:
            count+=1
            cluster = path.split(enrich_string)[1].replace('.txt', '')
            current_df = pd.read_csv(path, sep='\t')
            try:
                significant_terms = current_df[current_df['p_bonferroni']<p_threshold]
            except KeyError:
                continue
            if len(significant_terms) > 0:
                go_enrich_dict[cluster] = significant_terms[['id', 'description', 'p_bonferroni']]
        else:
            if '.txt' in path:
                try:
                    df = pd.read_csv(path, header=None)
                except:
                    print 'Nothing in %s' %path
                    continue
                if 'Spry2' in df.values:
                    print('Spry2 ', path)
                elif 'Spry4' in df.values:
                    print('Spry4 ', path)
                elif 'Spry3' in df.values:
                    print('Spry3', path)

    return go_enrich_dict

def assess_tf_enrichment(directory, p_threshold):
    pathiter = make_path_iter(directory)
    tf_enrich_dict = {}
    count = 0
    enrich_string = "tf_enrich_"
    for path in pathiter:
        if 'whole' in path:
            continue
        if enrich_string in path:
            count+=1
            cluster = int(path.split(enrich_string)[1].replace('.txt', ''))
            current_df = pd.read_csv(path, sep='\t')
            significant_terms = current_df[current_df['p_bonferroni']<p_threshold]
            if len(significant_terms) > 0:
                tf_enrich_dict[cluster] = significant_terms[['TF', 'p_uncorrected', 'p_bonferroni']]

    return tf_enrich_dict

if __name__ == '__main__':
    #data_directory = 'Data/20150216'
    enrich_directory = 'clustering/go_enrichment/20151002_binary_clusters_0.05/WT/'
    pd.set_option('display.width',1000)
    p_threshold = 0.05
    c = assess_go_enrichment(enrich_directory, p_threshold)
    print(len(c.keys()))
    for key, value in c.iteritems():
        boring_terms = 0
        if 'biological_process' in value['description'].values:
            boring_terms+=1
        if 'cellular_component' in value['description'].values:
            boring_terms+=1
        if 'molecular_function' in value['description'].values:
            boring_terms+=1
        print
        if len(value['description'].values) > boring_terms:
            print(key)
            print(value)
            raw_input()