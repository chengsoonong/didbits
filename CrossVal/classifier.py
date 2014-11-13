"""Tools for interacting with the scikit-learn classifier."""

import gzip
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_and_predict(data_name, train_idx, pred_idx, method, C=None, sigma=None, verbose=False):
    """Work out which classifier to call on data_name.
    Then train the classifier, and predict on the validation set.

    return the accuracy of the classifier on the prediction set.
    """
    examples, labels = load_data(data_name)
    train_ex = examples[train_idx,:]
    pred_ex = examples[pred_idx,:]
    train_lab = labels[train_idx]
    pred_lab = labels[pred_idx]

    if method == 'rf':
        pred = _train_and_predict_sk_rf(train_ex, train_lab, pred_ex, verbose)
    elif method == 'svm':
        pred = _train_and_predict_sk_svm(train_ex, train_lab, pred_ex, sigma, C, verbose)
    else:
        raise NotImplementedError('Unexpected %s' % method)
    return accuracy(pred, pred_lab)

def load_data(data_name):
    """Load a gzipped csv file with labels in the first column"""
    data = np.genfromtxt(gzip.open(data_name+'.csv.gz'), delimiter=',')
    return data[:,1:], data[:,0]

def accuracy(output, labels):
    assert len(output) == len(labels)
    int_out = np.rint(np.sign(output))
    int_lab = np.rint(labels)
    match = len(np.nonzero(np.equal(int_out, int_lab))[0])
    return match/len(output)

def _train_and_predict_sk_rf(train_ex, train_lab, pred_ex, verbose):
    """Train and predict.
    Package: sklearn
    Algorithm: random forest
    """
    if verbose:
        print('Training sklearn random forest on %d examples with %d features'
              % (train_ex.shape))
    model = RandomForestClassifier()
    model.fit(train_ex, train_lab)
    if verbose:
        print('Predicting using sklearn random forest on %d examples with %d features'
              % (pred_ex.shape))
    pred = model.predict(pred_ex)
    return pred

def _train_and_predict_sk_svm(train_ex, train_lab, pred_ex, bandwidth, reg_param, verbose):
    """Train and predict.
    Package: sklearn
    Algorithm: support vector machine
    """
    if verbose:
        print('Training sklearn SVM on %d examples with %d features'
              % (train_ex.shape))
    frac_pos = (1.*np.sum(train_lab>0))/(1.*len(train_lab))
    model = SVC(C=reg_param, kernel='rbf', gamma=bandwidth, class_weight={1: 1./frac_pos})
    model.fit(train_ex, train_lab)
    if verbose:
        print('Predicting using sklearn SVM on %d examples with %d features'
              % (pred_ex.shape))
    pred = model.predict(pred_ex)
    return pred
    
