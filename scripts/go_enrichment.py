"""
This is based on code from the goatools team. It has been further functionalized to check multiple lists for enrichment

"""

"""
python %prog study.file population.file gene-association.file
This program returns P-values for functional enrichment in a cluster of study
genes using Fisher's exact test, and corrected for multiple testing (including
Bonferroni, Holm, Sidak, and false discovery rate).
About significance cutoff:
--alpha: test-wise alpha; for each GO term, what significance level to apply
        (most often you don't need to change this other than 0.05 or 0.01)
--pval: experiment-wise alpha; for the entire experiment, what significance
        level to apply after Bonferroni correction
"""

import sys
import os
import os.path as op
sys.path.insert(0, op.join(op.dirname(__file__), ".."))
from goatools import GOEnrichmentStudy
from goatools.obo_parser import GODag
import optparse

def read_geneset(study_fn, pop_fn, compare=False):
    pop = set(_.strip() for _ in open(pop_fn) if _.strip())
    study = frozenset(_.strip() for _ in open(study_fn) if _.strip())
    # some times the pop is a second group to compare, rather than the
    # population in that case, we need to make sure the overlapping terms
    # are removed first
    if compare:
        common = pop & study
        pop |= study
        pop -= common
        study -= common
        print(sys.stderr, "removed %d overlapping items" % (len(common)))
        print(sys.stderr, "Set 1: {0}, Set 2: {1}".format(len(study), len(pop)))

    return study, pop


def read_associations(assoc_fn):
    assoc = {}
    for row in open(assoc_fn):
        atoms = row.split()
        if len(atoms) == 2:
            a, b = atoms
        elif len(atoms) > 2 and row.count('\t') == 1:
            a, b = row.split("\t")
        else:
            continue
        b = set(b.split(";"))
        assoc[a] = b

    return assoc


def check_bad_args(args):
    """check args. otherwise if one of the 3 args is bad
    it's hard to tell which one"""
    import os
    if not len(args) == 3:
        return "please send in 3 file names"
    for arg in args[:-1]:
        if not os.path.exists(arg):
            return "*%s* does not exist" % arg

    return False

def check_enrichment(study_fn, pop_fn, assoc_fn, print_summary=False, save_summary = True, savepath=None, obo_dag=None):
    p = optparse.OptionParser(__doc__)

    p.add_option('--alpha', default=0.05, type="float",
                 help="Test-wise alpha for multiple testing "
                 "[default: %default]")
    p.add_option('--pval', default=None, type="float",
                 help="Family-wise alpha (whole experiment), only print out "
                 "Bonferroni p-value is less than this value. "
                 "[default: %default]")
    p.add_option('--compare', dest='compare', default=False,
                 action='store_true',
                 help="the population file as a comparison group. if this "
                 "flag is specified, the population is used as the study "
                 "plus the `population/comparison`")
    p.add_option('--ratio', dest='ratio', type='float', default=None,
                 help="only show values where the difference between study "
                 "and population ratios is greater than this. useful for "
                 "excluding GO categories with small differences, but "
                 "containing large numbers of genes. should be a value "
                 "between 1 and 2. ")
    p.add_option('--fdr', dest='fdr', default=False,
                 action='store_true',
                 help="Calculate the false discovery rate (alt. to the "
                 "Bonferroni but slower)")
    p.add_option('--indent', dest='indent', default=False,
                 action='store_true', help="indent GO terms")

    (opts, args) = p.parse_args()
    args = [study_fn, pop_fn, assoc_fn]
    bad = check_bad_args(args)
    if bad:
        print(bad)
        sys.exit(p.print_help())

    min_ratio = opts.ratio
    if min_ratio is not None:
        assert 1 <= min_ratio <= 2

    assert 0 < opts.alpha < 1, "Test-wise alpha must fall between (0, 1)"

    study_fn, pop_fn, assoc_fn = args
    study, pop = read_geneset(study_fn, pop_fn, compare=opts.compare)
    assoc = read_associations(assoc_fn)
    methods = ["bonferroni", "sidak", "holm"]
    if opts.fdr:
        methods.append("fdr")
    if obo_dag is None:
        obo_file = "go-basic.obo"
        obo_dag = GODag(obo_file=obo_file)
    g = GOEnrichmentStudy(pop, assoc, obo_dag, alpha=opts.alpha, methods=methods)

    results = g.run_study(study)

    if print_summary:
        g.print_summary(results, min_ratio=min_ratio, indent=opts.indent, pval=opts.pval)

    if save_summary:
        if savepath is None:
            savepath = study_fn.replace(study_fn.split("/")[-1], "enrichment_"+study_fn.split("/")[-1])
        g.wr_tsv(savepath, results)


def cluster_enrichment(study_basepath, population_path, association_path, savefolder=None, obo_dag=None):
    pathiter = (os.path.join(root, filename) for root, _, filenames in os.walk(study_basepath) for filename in
                filenames)
    for path in pathiter:
        if path == population_path or "population" in path or "enrichment_" in path:
            continue
        if savefolder is None:
            save_name = path.replace(path.split("/")[-1], "enrichment_"+path.split("/")[-1])
        else:
            save_name = savefolder+"enrichment_"+path.split("/")[-1]
        if obo_dag is None:
            obo_file = "go-basic.obo"
            obo_dag = GODag(obo_file=obo_file)
        check_enrichment(path, population_path, association_path, savepath=save_name, obo_dag=obo_dag)

if __name__ == "__main__":

    """
    studypath = '../clustering/go_enrichment/biclusters_ko_to_wt/'
    population = "../data/goa_data/mus_musculus_gene_set"
    association = "../data/goa_data/gene_association"
    obo = '../data/goa_data/go-basic.obo'
    obo_dag = GODag(obo_file=obo)
    cluster_enrichment(studypath, population, association, studypath+'go_mus_musculus_set/', obo_dag)
    sys.exit()
    """

    study = "../pipelines/GSE69822/go_enrich/pten_drg_list.txt"
    population = "../pipelines/GSE69822/go_enrich/population.txt"
    association = "../data/goa_data/human_go_associations.txt"
    obo = '../data/goa_data/go-basic.obo'
    obo_dag = GODag(obo_file=obo)
    check_enrichment(study, population, association, obo_dag=obo_dag,
                     save_summary=True, savepath='../pipelines/GSE69822/go_enrich/pten_drg_enrich.tsv')
    sys.exit()