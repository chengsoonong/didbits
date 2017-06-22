"""Classes to encapsulate the idea of a dataset in machine learning,
   including file access. Similar to mldata.py.
   The file format is a pytables zlib compressed hdf5 file.

   A dataset is modeled as an (example,label) tuple, each of which is an array.
   The idea is to avoid loading the whole dataset into memory.
"""

from os.path import isfile
import numpy
from numpy import array, unique
from numpy.random import permutation
from tables import openFile


class Dataset(object):
    """Encapsulate the data as well as permutations.
    This is to ensure consistent splitting of training and test sets.
    """
    def __init__(self, name, data_file=None, perm_file=None,
                 data_dir='', frac_train=0.7):
        """Initialize file names and parameters"""
        self.name = name
        self.data_dir = data_dir
        self.frac_train = frac_train
        if data_file:
            self.filename = data_file
        else:
            self.filename = '%s/%s_mat.h5' % (self.data_dir, name)
        self.data = openFile(self.filename, 'r')
        self.examples = self.data.root.examples
        self.labels = self.data.root.labels
        self.num_class = len(unique(array(self.labels)))

        self._get_set_permfile(perm_file)

    @property
    def num_examples(self):
        return self.examples.shape[1]

    @property
    def num_features(self):
        return self.examples.shape[0]

    def _generate_perm(self, num_perm):
        """Generate permutations of the index of the examples"""
        self.perms = []
        for iperm in range(num_perm):
            self.perms.append(permutation(self.num_examples))
        self.perms = array(self.perms)

    def _get_set_permfile(self, perm_file, num_perm=50):
        """If a permutation file exists, load it.
        Otherwise, generate it and save one.

        Sets self.perm_filename and self.perms
        """
        if perm_file:
            self.perm_filename = perm_file
        else:
            self.perm_filename = '%s/%s_perm.txt' % (self.data_dir, self.name)

        if isfile(self.perm_filename):
            self.perms = numpy.loadtxt(self.perm_filename, dtype=int, delimiter=' ')
        else:
            self._generate_perm(num_perm)
            fp = open(self.perm_filename, 'w')
            for iperm in range(num_perm):
                cur_mix = permutation(self.num_examples)
                fp.write(' '.join(map(str, cur_mix.tolist())))
                fp.write('\n')
            fp.close()

    def get_id_str(self, split_idx, split_type, fold=None, num_cv=None):
        self.perm_idx = split_idx
        return self._param2string(split_idx, split_type, fold, num_cv)

    def get_perm(self, split_idx, split_type, fold=None, num_cv=None):
        """Returns the indices of the training and test examples."""
        assert(split_type == 'val' or split_type == 'test')
        self.perm_idx = split_idx
        perm = self.perms[split_idx]
        split_train = int(self.frac_train*self.num_examples)
        if split_type == 'val':
            perm_train = perm[:split_train]
            idx_pred = perm_train[fold::num_cv]
            idx_train = []
            for idx in perm_train:
                if not (idx in idx_pred):
                    idx_train.append(idx)
            idx_train = numpy.array(idx_train)

        elif split_type == 'test':
            idx_train = perm[:split_train]
            idx_pred = perm[split_train:]

        id_str = self._param2string(split_idx, split_type, fold, num_cv)
        return (idx_train, idx_pred, id_str)

    def close(self):
        """Close the HDF5 file"""
        self.data.close()

    def _param2string(self, split_idx, split_type, fold, num_cv):
        """Generate part of file name the split"""
        return param2string(split_idx, split_type, fold, num_cv, self.frac_train)


def param2string(split_idx, split_type, fold=None, num_cv=None, frac_train=None):
    """Generate part of file name the split"""
    if split_type == 'test':
        id_str = '%d_r%1.2f' % (split_idx, float(frac_train))
    elif split_type == 'val':
        id_str = '%d_cv%d:%d' % (split_idx, fold, num_cv)
    return id_str
