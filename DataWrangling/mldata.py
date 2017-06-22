#!/usr/bin/env python

"""Classes to encapsulate the idea of a dataset in machine learning,
   including file access. Currently this focuses on reading and writing
   transparently to different file formats.

   A dataset is modeled as an (example,label) tuple, each of which is an array.
   The base class doesn't know how to split, so just returns one array.

   The classes currently implemented use three
   different ways of iterating through files:
   - CSV uses the python module csv's iterator
   - Libsvm uses a hand crafted while loop
   - ARFF always reads the whole file, and does a slice
   - FASTA uses a hand crafted while loop that behaves like a generator

   The class DatasetFileARFF is in mldata-arff.py.
"""

import numpy
from numpy import array, concatenate, unique
from numpy.random import permutation
import csv
from io import FileIO as BUILTIN_FILE_TYPE


class DatasetFileBase(BUILTIN_FILE_TYPE):
    """A Base class defining barebones and common behaviour
    """

    def __init__(self, filename, extype):
        """Just the normal file __init__,
        followed by the specific class corresponding to the file extension.

        """
        self.extype = extype
        self.filename = filename

    def readlines(self, idx=None):
        """Read the lines defined by idx (a numpy array).
        Default is read all lines.

        """
        if idx is None:
            data = self.readlines()
        else:
            data = self.readlines()[idx]
        return data

    def writelines(self, data, idx=None):
        """Write the lines defined by idx (a numpy array).
        Default is write all lines.

        data is assumed to be a numpy array.

        """
        if idx is None:
            self.writelines(data)
        else:
            self.writelines(data[idx])


class DatasetFileCSV(DatasetFileBase):
    """Comma Seperated Values file.

    Labels are in the first column.

    """
    def __init__(self, filename, extype, verbose=False):
        self.verbose = verbose
        DatasetFileBase.__init__(self, filename, extype)

    def readlines(self, idx=None):
        """Read from file and split data into examples and labels"""
        reader = csv.reader(open(self.filename, 'r'), delimiter=',', quoting=csv.QUOTE_NONE)
        labels = []
        examples = []
        for ix, line in enumerate(reader):
            if idx is None or ix in idx:
                labels.append(float(line[0]))
                if self.extype == 'vec':
                    examples.append(array(list(map(float, line[1:]))))
                elif self.extype == 'seq':
                    examples.append(line[1:][0])
                elif self.extype == 'mseq':
                    examples.append(array(line[1:]))

        if self.extype == 'vec':
            examples = array(examples).T
            if self.verbose:
                print(('%d features, %d examples' % examples.shape))
        elif self.extype == 'seq':
            if self.verbose:
                print(('sequence length = %d, %d examples' % (len(examples[0]), len(examples))))
        elif self.extype == 'mseq':
            printstr = 'sequence lengths = '
            for seq in examples[0]:
                printstr += '%d, ' % len(seq)
            printstr += '%d examples' % len(examples)
            if self.verbose:
                print(printstr)
        return (examples, array(labels))

    def writelines(self, examples, labels, idx=None):
        """Merge the examples and labels and write to file"""
        if idx is None:
            idx = list(range(len(labels)))
        if self.extype == 'seq':
            data = list(zip(labels[idx], list(array(examples)[idx])))
        if self.extype == 'mseq':
            data = []
            for ix, curlab in enumerate(labels):
                data.append([curlab]+list(examples[ix]))
        elif self.extype == 'vec':
            data = []
            for ix, curlab in enumerate(labels):
                data.append(concatenate((array([curlab]), examples[:, ix].T)))

        fp = open(self.filename, 'w')
        writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE)
        for ix in idx:
            writer.writerow(data[ix])
        fp.close()


class DatasetFileLibsvm(DatasetFileBase):
    """LibSVM format, where the first column is assumed to be the label,
    and the entries are in sparse matrix format."""
    def __init__(self, filename, extype='vec', verbose=False):
        assert(extype == 'vec')
        self.verbose = verbose
        DatasetFileBase.__init__(self, filename, extype)

    def readlines(self, idx=None):
        """Read from file and split data into examples and labels"""
        if idx is not None:
            raise NotImplementedError
        infile = open(self.filename, 'r')
        num_dim = 0
        parsed = []
        for line in infile:
            (pline, max_idx) = self._parse_line(line)
            parsed.append(pline)
            num_dim = max(num_dim, max_idx)
        infile.close()

        labels = []
        if self.verbose:
            print(('%d features, %d examples' % (num_dim, len(parsed))))
        examples = numpy.zeros((int(num_dim), len(parsed)))
        for ix in range(len(parsed)):
            labels.append(parsed[ix]['label'])
            for v in parsed[ix]['variables']:
                examples[int(v[0])-1, ix] = float(v[1])
        return (examples, array(labels))

    def _parse_line(self, line):
        """Parse a LibSVM input line and return attributes."""
        items = line.split()
        label = int(items[0])-1
        variables = []
        max_idx = 0
        for entry in items[1:]:
            val = entry.split(':')
            max_idx = max(max_idx, int(val[0]))
            variables.append((int(val[0]), float(val[1])))
        return ({'label': label, 'variables': variables}, max_idx)

    def writelines(self, examples, labels, idx=None):
        """Merge the examples and labels and write to file"""
        if idx is None:
            idx = list(range(len(labels)))

        num_feat = examples.shape[0]
        fp = open(self.filename, 'w')
        for ix in idx:
            fp.write('%d ' % (labels[ix]+1))
            for ix_feat in range(num_feat):
                if examples[ix_feat, ix] != 0.0:
                    fp.write('%d:%f ' % (ix_feat + 1, examples[ix_feat, ix]))
            fp.write('\n')
        fp.close()


class DatasetFileFASTA(DatasetFileBase):
    """Fasta format file, labels are in the comment after keyword 'label'.
    label=1
    label=-1

    """
    def __init__(self, filename, extype):
        if extype != 'seq':
            print('Can only write fasta file for sequences!')
            raise IOError
        DatasetFileBase.__init__(self, filename, extype)
        self.fp = None

    def readlines(self, idx=None):
        """Read from file and split data into examples and labels"""
        self.fp = open(self.filename, 'r')
        line = self.fp.readline()

        examples = []
        labels = []
        ix = 0
        while True:
            if not line:
                break
            (ex, lab, line) = self.readline(line)
            if idx is None or ix in idx:
                examples.append(ex)
                labels.append(lab)
            ix += 1
        self.fp.close()
        print(('sequence length = %d, %d examples' % (len(examples[0]), len(examples))))
        return (examples, array(labels))

    def writelines(self, examples, labels, idx=None, linelen=60):
        """Write the examples and labels and write to file"""
        if idx is None:
            idx = list(range(len(labels)))

        fp = open(self.filename, 'w')
        for ix in idx:
            fp.write('> %d label=%d\n' % (ix, round(labels[ix])))
            for lineidx in range(0, len(examples[ix]), linelen):
                fp.write(examples[ix][lineidx:lineidx+linelen] + '\n')
        fp.close()

    def readline(self, line):
        """Reads a fasta entry and returns the label and the sequence"""
        if line[0] == '':
            return

        assert(line[0] == '>')
        # Use list comprehension to get the integer that comes after label=
        a = line.split()
        label = float([b.split('=')[1] for b in a if b.split('=')[0] == 'label'][0])

        lines = []
        line = self.fp.readline()
        while True:
            if not line:
                break
            if line[0] == ">":
                break
            # Remove trailing whitespace, and any internal spaces
            lines.append(line.rstrip().replace(" ", ""))
            line = self.fp.readline()

        return (''.join(lines), label, line)


def init_datasetfile(filename, extype):
    """A factory that returns the appropriate class based on the file extension.

    recognised file extensions
    - .csv  : Comma Separated Values
    - .tab  : General tabular data
    - .arff : Attribute-Relation File Format (weka)
    - .fa   : Fasta file format (seq only)
    - .fasta: same as above.

    Since the file type does not determine what type of data is actually being used,
    the user has to supply the example type.

    extype can be ('vec','seq','mseq')
    vec - array of floats
    seq - single sequence
    mseq - multiple sequences

    """
    allowedtypes = ('vec', 'seq', 'mseq')
    assert(extype in allowedtypes)
    # map the file extensions to the relevant classes
    _format2dataset = {'csv': DatasetFileCSV,
                       'fa': DatasetFileFASTA,
                       'fasta': DatasetFileFASTA,
                       'libsvm': DatasetFileLibsvm,
                       }

    extension = detect_extension(filename)
    return _format2dataset[extension](filename, extype)


def detect_extension(filename):
    """Get the file extension"""
    if filename.count('.') > 1:
        print('WARNING: %s has more than one . using last one' % filename)
    detect_ext = filename.split('.')[-1]
    known_ext = ['csv', 'fasta', 'fa', 'libsvm']

    if detect_ext not in known_ext:
        print(('WARNING: %s is an unknown file extension, defaulting to csv' % detect_ext))
        detect_ext = 'csv'

    return detect_ext


def convert(infile, outfile, extype):
    """Copy data from infile to outfile, possibly converting the file format."""
    fp1 = init_datasetfile(infile, extype)
    (examples, labels) = fp1.readlines()
    fp2 = init_datasetfile(outfile, extype)
    fp2.writelines(examples, labels)


class DatasetBase(object):
    """Encapsulate the data as well as permutations.
    This is to ensure consistent splitting of training and test sets.
    """
    def __init__(self):
        self.name = ''

        self.filename = ''
        self.examples = None
        self.labels = None
        self.num_class = -1

        self.perm_filename = ''
        self.perms = numpy.array([])  # all the permutations
        self.perm_idx = -1

        self.data_dir = ''
        self.frac_train = 0.7

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

    def generate_perm(self, num_perm=50):
        """Generate permutations of the index of the examples"""
        self.perms = []
        for iperm in range(num_perm):
            self.perms.append(permutation(self.num_examples))
        self.perms = array(self.perms)

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


class Dataset(DatasetBase):
    """Encapsulate the data as well as permutations.
    This is to ensure consistent splitting of training and test sets.

    Initialize dataset from a file.
    """
    def __init__(self, name, data_file=None, perm_file=None,
                 data_dir='', frac_train=0.7, read_data=True):
        """Load the data into memory"""
        self.name = name
        self.data_dir = data_dir
        self.frac_train = frac_train
        if data_file:
            self.filename = data_file
        else:
            self.filename = '%s/%s.csv' % (self.data_dir, name)
        data = init_datasetfile(self.filename, 'vec')
        if read_data:
            (self.examples, self.labels) = data.readlines()
            self.num_class = len(unique(self.labels))

        if perm_file:
            self.perm_filename = perm_file
        else:
            self.perm_filename = '%s/%s_perm.txt' % (self.data_dir, name)
        if read_data:
            self.perms = numpy.loadtxt(self.perm_filename, dtype=int, delimiter=' ')

    @property
    def num_examples(self):
        return self.examples.shape[1]

    @property
    def num_features(self):
        return self.examples.shape[0]


def generate_perm(datafile, permfile, num_perm=50):
    """Read in the data in datafile, and generate a file
    with permutations of the index of the examples.
    """
    fp1 = init_datasetfile(datafile, 'vec')
    (examples, labels) = fp1.readlines()
    num_examples = len(labels)
    fp2 = open(permfile, 'w')
    for iperm in range(num_perm):
        cur_mix = permutation(num_examples)
        fp2.write(' '.join(map(str, cur_mix.tolist())))
        fp2.write('\n')
    fp2.close()
