from unittest import TestCase

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
