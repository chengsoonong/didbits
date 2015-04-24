from numpy import sign, nonzero, equal, zeros, mean, std
from numpy import array, concatenate, sqrt, diag, matrix
from numpy import log, pi, trace, logspace, log10
from numpy.linalg import det, pinv
from numpy import sum, arange, nan, unique, argsort, isnan, cumsum
from scipy import stats


def confusionMatrix(labels_test, labels_predicted):
    """Compute the matrix of predictions versus labels"""
    if len(labels_test) != len(labels_predicted):
        return 0
    TP = 0; FP = 0; TN = 0; FN = 0
    for i in range(0, len(labels_test)):
        if labels_test[i] == 0 or labels_predicted[i] == 0:
            return 0
        if labels_test[i] > 0:
            if labels_predicted[i] > 0: TP += 1
            else: FN +=1
        else:
            if labels_predicted[i] > 0: FP += 1
            else: TN += 1
    return (TP, TN, FP, FN)

def accuracy(output, labels_test):
    """How many correct predictions?"""
    if max(labels_test) > 1:
        return accuracy_multiclass(output, labels_test)
    else:
        TP, TN, FP, FN = confusionMatrix(labels_test, sign(output))
        return float(TP + TN) / (TP + TN + FP + FN)

def balanced_accuracy(output, labels):
    """How many correct predictions?, normalized by number of positives and negatives"""
    assert all(unique(labels)==array([-1,1])), 'Binary classification only'
    TP, TN, FP, FN = confusionMatrix(labels, sign(output))
    return 0.5*TP/(TP+FN) + 0.5*TN/(FP+TN)

def balanced_accuracy_multitask(output, labels):
    """Balanced accuracy applied to each task separately"""
    assert output.shape == labels.shape, 'Predictions and labels have different shape'
    num_task = output.shape[0]
    balacc = zeros(num_task)
    for ix in range(num_task):
        lab_idx = nonzero(labels[ix,:])
        balacc[ix] = balanced_accuracy(output[ix,lab_idx].flatten(), labels[ix,lab_idx].flatten())
    return balacc

def accuracy_multiclass(output, labels_test):
    """Multiclass accuracy"""
    int_out = map(int, output)
    int_lab = map(int, labels_test)
    return 1.0*len(nonzero(equal(int_out, int_lab))[0])/len(output)

def rmse(output, labels):
    """Root mean squared error"""
    if output.ndim == 1:
        num_ex = len(output)
    else:
        num_ex = output.shape[1]
    return sqrt(diag(matrix(output - labels)*matrix(output-labels).T))/num_ex

def rmse_multitask(output, labels):
    """rmse for each task separately"""
    assert output.shape == labels.shape, 'Predictions and labels have different shape'
    num_task = output.shape[0]
    error = zeros(num_task)
    for ix in range(num_task):
        lab_idx = nonzero(labels[ix,:])
        error[ix] = rmse(output[ix,lab_idx].flatten(), labels[ix,lab_idx].flatten())
    return error

def differential_entropy(K):
    """The differential entropy for a multivariate normal distribution.
    Assume K is a positive semidefinite matrix."""
    d = K.shape[0]
    return (d/2)*(1+log(2*pi)) + 0.5*log(det(K))

def relative_entropy(K):
    """The relative entropy for a multivariate normal distribution,
    compared to another Gaussian with the same mean and identity covariance.
    """
    d = K.shape[0]
    return 0.5*(log(det(pinv(K))) + trace(K) - d)

def trapz(x, y, ylim0, ylim1):
    """Trapezoidal rule for integrating
    the curve defined by x-y pairs.
    Assume x is in the range [0,1]
    and y is constant to the ends of the interval
    """
    assert len(x) == len(y), 'x and y need to be of same length'
    x = concatenate([x, array([0.0, 1.0])])
    y = concatenate([y, array([ylim0, ylim1])])
    sort_idx = argsort(x)
    sx = x[sort_idx]
    sy = y[sort_idx]
    area = 0.0
    for ix in range(len(x)-1):
        area += 0.5*(sx[ix+1]-sx[ix])*(sy[ix+1]+sy[ix])
    return area

def stats_empirical(output, labels, interp=1000):
    """Compute some statistics for binary predictions.
    tpr - true positive rate (recall)
    fpr - false positive rate
    ppv - positive predictive value (precision)
    auc - area under the ROC curve
    ap - area under the precision-recall curve (average precision)

    If there are more than interp=1000 number of examples, then compute in logspace intervals
    """
    assert len(output)==len(labels), 'Predictions and labels have different lengths'
    assert all(unique(labels)==array([-1,1])), 'Labels are not binary {-1,+1}'

    # Sort true labels according to predictions in ascending order
    n = len(output)
    sort_idx = argsort(output)
    sorted_labels = labels[sort_idx]

    tpr = []
    fpr = []
    ppv = []
    if n > interp:
        thresholds = list(range(100))+list(map(int, logspace(2, log10(n), interp).round()))
        thresholds = (n-array(thresholds))[::-1]
    else:
        thresholds = range(n+1)
    for thres in thresholds:
        tp = 1.0*sum(sorted_labels[thres:]>0)
        fn = 1.0*sum(sorted_labels[:thres]>0)
        tn = 1.0*sum(sorted_labels[:thres]<0)
        fp = 1.0*sum(sorted_labels[thres:]<0)

        if tp+fn > 0.0:
            tpr.append(tp/(tp+fn))
        else:
            tpr.append(nan)
        if fp+tn > 0.0:
            fpr.append(fp/(fp+tn))
        else:
            fpr.append(nan)
        if tp+fp > 0.0:
            ppv.append(tp/(tp+fp))
        else:
            ppv.append(nan)
    
    tpr = array(tpr)
    fpr = array(fpr)
    ppv = array(ppv)

    auc = trapz(fpr, tpr, 0.0, 1.0)
    idx = -isnan(ppv)
    apr = trapz(tpr[idx], ppv[idx], 1.0, 0.0)

    return tpr, fpr, ppv, auc, apr

def stats_binormal(output, labels, step=0.001):
    """Compute some statistics for binary predictions.
    tpr - true positive rate (recall)
    fpr - false positive rate
    ppv - positive predictive value (precision)
    auc - area under the ROC curve
    ap - area under the precision-recall curve (average precision)

    Use the binormal assumption.
    step gives the smoothness of curve.
    """
    assert len(output)==len(labels), 'Predictions and labels have different lengths'
    assert all(unique(labels)==array([-1,1])), 'Labels are not binary {-1,+1}'

    # Estimate the binormal parameters
    pos = output[labels>0]
    neg = output[labels<0]
    mu_pos = mean(pos)
    mu_neg = mean(neg)
    std_pos = std(pos)
    std_neg = std(neg)
    alpha = 1.0*len(pos)/len(output)

    # Sanity checks
    assert mu_pos > mu_neg, 'positive Gaussian is not to the right of negative'
    assert (std_pos>0) and (std_neg>0), 'Variance is zero'

    # Use Gaussian cdf to estimate scores
    thres = arange(mu_neg-5.0*std_neg, mu_pos+5*std_pos, step)
    tp = alpha*(1.0-stats.norm.cdf(thres, mu_pos, std_pos))
    fp = (1.0-alpha)*(1.0-stats.norm.cdf(thres, mu_neg, std_neg))
    fn = alpha*stats.norm.cdf(thres, mu_pos, std_pos)
    tn = (1.0-alpha)*stats.norm.cdf(thres, mu_neg, std_neg)

    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    ppv = tp/(tp+fp)
    A = (mu_pos-mu_neg)/std_pos
    B = std_neg/std_pos
    auc = stats.norm.cdf(A/sqrt(1+B*B))
    apr = trapz(tpr, ppv, 1.0, 0.0)

    return tpr, fpr, ppv, auc, apr

def stats_binormal_multitask(output, labels):
    """stats_binormal applied to each row"""
    assert output.shape == labels.shape, 'Predictions and labels have different shape'
    num_task = output.shape[0]
    tpr = []
    fpr = []
    ppv = []
    auc = zeros(num_task)
    apr = zeros(num_task)
    for ix in range(num_task):
        lab_idx = nonzero(labels[ix,:])
        ctpr, cfpr, cppv, auc[ix], apr[ix] = stats_binormal(output[ix,lab_idx].flatten(),
                                                            labels[ix,lab_idx].flatten())
        tpr.append(ctpr)
        fpr.append(cfpr)
        ppv.append(cppv)
    return tpr, fpr, ppv, auc, apr

def auc(output, labels):
    """The area under the ROC curve,
    estimated using the binormal approximation
    """
    tpr, fpr, ppv, auc, apr = stats_empirical(output, labels)
    return auc

def r2(output, labels):
    """The squared correlation coefficient"""
    mu = mean(output)
    numerator = sum((labels-output)*(labels-output))
    denominator = sum((labels-mu)*(labels-mu))
    return 1.-(numerator/denominator)

def spearman(output, labels):
    """Spearman's correlation coefficient (rho)"""
    output_rank = score2rank(output)
    labels_rank = score2rank(labels)
    rho, pval = stats.pearsonr(output_rank, labels_rank)
    return rho


def score2rank(orig_scores):
    """Convert an array of scores into an array of normalised ranks,
    such that the highest score has the highest rank (1/num_ex)."""
    scores = orig_scores.copy()
    idx_sort = argsort(scores)[::-1]
    unsort = argsort(idx_sort)
    scores = scores[idx_sort]

    ranks = infer_ranks(scores)
    
    assert(len(ranks) == len(scores))
    ranks = ranks/(len(scores)+1.0)
    ranks = ranks[unsort]
    return ranks

