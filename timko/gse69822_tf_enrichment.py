import multiprocessing as mp

import pandas as pd
from pydiffexp.utils import fisher_test as ft


def tf_work(g: ft.Enricher, genes, label):
    print("Enrich search", label)
    enrich = g.study_enrichment(genes)
    enrich = enrich[enrich.p_bonferroni < 0.05].copy()
    enrich.index.name = label
    print(len(enrich))
    return enrich


def tf_cluster_search(gene_sets: dict, gene_to_tf: dict, background=None):
    """

    :param gene_sets: dict; clusters (keys) and gene list in the cluster (values)
    :param gene_to_tf: dict; gene (keys) and associated transcription factor lists (values)
    :param background: list-like; genes to use as the background in fisher's exact test
    :return: DataFrame; Multiindex DataFrame with cluster and enriched Transcription Factors as the index
    """
    er = ft.Enricher(gene_to_tf, background=background)

    sets = [(er, frozenset(s), l) for l, s in gene_sets.items()]
    labels = [ss[2] for ss in sets]
    pool = mp.pool.ThreadPool()
    df_list = pool.starmap(tf_work, sets)
    pool.close()
    pool.join()

    # Pandas multiindex workaround
    results = pd.DataFrame()
    for lab, ddf in zip(labels, df_list):
        if len(ddf):
            ddf.index.name = 0
            ddf = ddf.reset_index()
            lab = lab if isinstance(lab, tuple) else tuple([lab])
            ddf['clust'] = [lab] * len(ddf)
            ddf = ddf.set_index(['clust', 0])
            results = pd.concat([results, ddf])

    return results

def main():
    """
     ===================================
     ====== Set script parameters ======
     ===================================
     """
    # todo: argv and parsing

    # Set globals
    pd.set_option('display.width', 250)

    # External files
    gene_to_tf_dict = pd.read_pickle('human_encode_associations.pkl')

    # Load the data
    gene_sets = pd.read_pickle('gene_sets.pkl')
    background_genes = gene_to_tf_dict.keys()

    # Single enrichment search
    e = ft.Enricher(gene_to_tf_dict)
    enriched = e.study_enrichment(gene_sets[(0, 0, 0, 0, 0, 1)], bg_genes=background_genes)
    print(enriched.head())

    # Parallelized search that doesn't work yet
    tf_enriched = tf_cluster_search(gene_sets, gene_to_tf_dict, background=background_genes)
    print(tf_enriched.head())


if __name__ == '__main__':
    main()
