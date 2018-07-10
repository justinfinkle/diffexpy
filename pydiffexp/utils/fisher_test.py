__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def make_association_dict(association_file, term='TF', association='Gene', sep='\t'):
    """
    Read an association file and turn it into a dictionary
    :param association_file:
    :param term:
    :param association:
    :param sep:
    :return:
    """
    # Read into a dataframe
    association_df = pd.read_csv(association_file, sep=sep)

    term_dict = {term.upper(): df_subset[association].tolist() for term, df_subset in association_df.groupby(term)}

    # todo: this needs to be generalized
    association_dict = {gene.split(';')[0].upper(): df_subset["TF"].tolist() for gene, df_subset in
                        association_df.groupby("Gene")}

    return term_dict, association_dict


def convert_gene_to_tf(gene_list, gene_dict):
    # todo: this needs to be generalized
    """
    Convert a list of genes to a list of transcription factors
    :param gene_list:
    :param gene_dict:
    :return:
    """
    gene_list = [g.upper() for g in gene_list]
    tf_list = []
    count = 0
    for gene in gene_list:
        if gene in gene_dict.keys():
            count += 1
            tf_list = tf_list+gene_dict[gene]
        else:
            continue

    tf_dict = {tf: [] for tf in tf_list}

    # Add genes associated with each tf
    for k in tf_dict.keys():
        for gene in gene_list:
            if gene in gene_dict.keys():
                if k in gene_dict[gene]:
                    tf_dict[k].append(gene)
                else:
                    continue
            else:
                continue

    return tf_list, tf_dict


def calculate_study_enrichment(study_term_list, background_term_list, fdr=0.05):
    """
    Calculate the enrichment of terms in the study
    :param study_term_list: list
        Terms found in the study
    :param background_term_list: list
        Terms found in the background
    :param fdr: float
        The expected false discovery rate for which to correct
    :return: dataframe
    """
    # List of unique terms in the study
    study_term_set = list(set(study_term_list))

    # Calculate p_value for each term
    p_values = np.array([fisher_score(tf, study_term_list, background_term_list) for tf in study_term_set])

    # Bonferroni correction
    corrected_p = p_values*len(study_term_set)

    # Make results table
    results_table = pd.DataFrame(np.vstack((p_values, corrected_p)).T, columns=['p_uncorrected', 'p_bonferroni'])
    results_table.insert(0, 'TF', study_term_set)
    results_table.sort_values('p_uncorrected', inplace=True)

    #FDR correcton
    results_table['FDR_thresh'] = np.arange(1, len(results_table)+1)/float(len(results_table))*fdr
    results_table['FDR_reject'] = results_table['p_uncorrected'] < results_table['FDR_thresh']
    results_table.sort_values('FDR_reject', ascending=False, inplace=True)
    return results_table


def fisher_score(x, study_list, background_list):
    """
    Get the fisher exact p-value of a term in a list compared to the background
    :param x: str
        The label that is being checked for enrichment in the study.
    :param study_list: list
        Terms found in the study
    :param background_list: list
        Terms found in the background
    :return: float
        The p-value associated with the term
    """
    c_table = make_contingency_table(x, study_list, background_list)

    # One sided test looking for over-representation so use 'greater'
    _, pvalue = fisher_exact(c_table, alternative='greater')
    return pvalue


def make_contingency_table(x, study_list, background_list):
    """
    Makes a 2x2 contingency table
    :param x: str
        The label that is being checked for enrichment in the study.
    :param study_list: list
        Terms found in the study
    :param background_list: list
        Terms found in the background
    :return: array
        The 2x2 contingency table used for calculating enrichment
    """
    # Initialize table
    contingency_table = np.zeros((2, 2))
    count_in_study = study_list.count(x)
    count_in_background = max(0, background_list.count(x)-count_in_study)
    contingency_table[0, 0] = count_in_study
    contingency_table[0, 1] = len(study_list) - count_in_study
    contingency_table[1, 0] = count_in_background
    contingency_table[1, 1] = len(background_list) - count_in_background
    return contingency_table
