from unittest import TestCase

import nose
from scipy import stats

from pydiffexp.utils import fisher_test as ft


class TestFisher(TestCase):
    def test_conversion(self):
        """
        The keys of the dictionary should be switched to sets of values
        :return:
        """
        in_dict = {'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [3, 4, 5]}
        in_list = list(in_dict.keys())
        # Add a value that isn't in the dictionary. This should not appear in the output
        not_in_dict = 'd'
        in_list.append(not_in_dict)
        expected_out = {1: {'a'}, 2: {'a', 'b'}, 3: {'a', 'b', 'c'},
                        4: {'b', 'c'}, 5: {'c'}}
        out = ft.tf_to_gene_dict(in_list, in_dict)

        # The dictionaries should be equal
        self.assertTrue(expected_out == out)

        # This is likely redundant, but this value shouldn't be in the keys
        self.assertTrue(not_in_dict not in out.keys())
        print('Passed conversion')

        return

    def test_tf_fisher(self):
        # Total of 16 genes
        n_genes = 16
        list_of_genes = ['G{}'.format(ii+1) for ii in range(n_genes)]
        study_genes = list_of_genes[:9]
        bg_genes = list(sorted(set(list_of_genes).difference(study_genes)))
        tfs = ['TF1', 'TF2']
        tf_assoc = {gene: [tfs[0]] if ii < 8 else [tfs[1]] for ii, gene in enumerate(study_genes)}
        for ii, gene in enumerate(bg_genes):
            tf_assoc[gene] = tfs[0] if ii < 2 else tfs[1]

        er = ft.Enricher(tf_assoc)
        e = er.study_enrichment(study_genes)

        c_tables = {'TF1': [[8, 2], [1, 5]],
                    'TF2': [[1, 5], [8, 2]]}

        for tf in tfs:
            _, expected = stats.fisher_exact(c_tables[tf], alternative='greater')
            actual = e.loc[tf, 'p_uncorrected']
            self.assertAlmostEqual(actual, expected)

        return


if __name__ == '__main__':
    nose.main()