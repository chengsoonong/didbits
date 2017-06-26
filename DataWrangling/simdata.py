"""Some functions to generate simulated data"""
try:
    from os import link
except ImportError:
    # Hack for windows
    from shutil import copy2

    def link(src, dst):
        copy2(src, dst)

from numpy.random import randn, rand, permutation, randint, seed
from numpy import where, nonzero, sqrt, real
from numpy import zeros, ones, eye, mean, kron, sign
from numpy import array, equal, argsort
from numpy import flatnonzero
from numpy.linalg import norm
from optwok.mldata import DatasetFileCSV
from optwok.testtools import flatten_list, list2matrix
from numpy import matrix, vstack, hstack, concatenate
from optwok.kernel import GaussKernel, JointKernel
from optwok.io_pickle import load, save
from optwok.mldata import DatasetBase
from scipy.linalg.matfuncs import sqrtm

#####################################################################
# Binary classification
#####################################################################


def cloudgen(numpoint, numfeat, numnoise, fracpos, width):
    """Generate two Gaussian point clouds, centered around one and minus one.
    Gaussian clouds are in numfeat dimensions.
    Generate uniform noise in numnoise dimensions.
    """
    numpos = int(round(fracpos*numpoint))
    numneg = numpoint - numpos

    metadata = 'cloudgen(%d,%d,%d,%d,%3.2f)' % (numpos, numneg, numfeat, numnoise, width)
    print(metadata)

    datapos = ones((numfeat, numpos)) + width*randn(numfeat, numpos)
    dataneg = -ones((numfeat, numneg)) + width*randn(numfeat, numneg)
    noise = (2.0+width)*(rand(numnoise, numpos+numneg)
                         - 0.5 * ones((numnoise, numpos+numneg)))
    pointcloud = 0.2*concatenate((concatenate((datapos, dataneg), axis=1),
                                  noise), axis=0)
    labels = concatenate((ones(numpos), -ones(numneg)))

    return metadata, pointcloud, labels


def cloudgen_weak(numpoint, numfeat, numweak, fracpos, width):
    """Generate two Gaussian point clouds, centered around one and minus one.
    Gaussian clouds are in numfeat dimensions.
    Overlapping Gaussian clouds are in numweak dimensions.
    """
    numpos = int(round(fracpos*numpoint))
    numneg = numpoint - numpos

    metadata = 'cloudgen_weak(%d,%d,%d,%d,%3.2f)' %\
               (numpos, numneg, numfeat, numweak, width)
    print(metadata)

    strongpos = ones((numfeat, numpos)) + width*randn(numfeat, numpos)
    strongneg = -ones((numfeat, numneg)) + width*randn(numfeat, numneg)
    weakpos = 0.1*ones((numweak, numpos)) + 2.0*width*randn(numweak, numpos)
    weakneg = -0.1*ones((numweak, numneg)) + 2.0*width*randn(numweak, numneg)
    datapos = concatenate((strongpos, weakpos), axis=0)
    dataneg = concatenate((strongneg, weakneg), axis=0)
    pointcloud = 0.2*concatenate((datapos, dataneg), axis=1)
    labels = concatenate((ones(numpos), -ones(numneg)))

    return metadata, pointcloud, labels


def create_data(num_train=500, num_test=200, num_feat=10, num_noise_feat=10, width=0.8,
                frac_pos=0.5, frac_flip=0.0, symm_flip=True):
    """Create mixture of Gaussians distribution"""
    (metadata, train_ex, train_labels) = cloudgen(num_train, num_feat,
                                                  num_noise_feat, frac_pos, width)
    (metadata, test_ex, test_labels) = cloudgen(num_test, num_feat,
                                                num_noise_feat, frac_pos, width)
    random_classification_noise(train_labels, frac_flip, symm_flip)

    return (train_ex, train_labels, test_ex, test_labels)


def _get_hyperplane(num_feat, in_simplex, homogeneous, min_val=0.01):
    """Randomly create a hyperplane"""
    if in_simplex:
        w = rand(num_feat, 1)
        w += min_val+1.0/num_feat
        w = w/sum(abs(w))
        if homogeneous:
            b = 0.0
        else:
            b = rand()
    else:
        w = rand(num_feat, 1) - 0.5
        w = sign(w)*(abs(w)+min_val+1.0/num_feat)
        w = w/norm(w)
        if homogeneous:
            b = 0.0
        else:
            b = 2.0*(rand()-0.5)
    return w, b


def _get_hyperplane_sparse(num_feat, num_nz, homogeneous, min_val=0.01):
    """Randomly create a hyperplane"""
    w = rand(num_feat, 1)-0.5
    w = sign(w)*(abs(w)+min_val+1.0/num_feat)
    w /= norm(w)
    small_idx = argsort(abs(w[:, 0]))[:(num_feat-num_nz)]
    w[small_idx] = 0.0
    if homogeneous:
        b = 0.0
    else:
        b = 2.0*(rand()-0.5)
    return w, b


def linear_separation(num_point, num_feat, num_noise, margin=0.5,
                      sparse=0, homogeneous=True, in_simplex=False):
    """Generate uniformly distributed data in [-1,1]^d,
    pick a random hyperplane and create labels according to it.
    """
    assert(margin < 1.0 and margin >= 0.0)
    margin /= float(num_feat)
    if sparse > 0:
        w, b = _get_hyperplane_sparse(num_feat, sparse, homogeneous)
        param = 'sparse_linear_separation(%d, %d, %d, %d, %1.2f, %d)' %\
                (num_point, num_feat, sparse, num_noise, margin, int(homogeneous))
    else:
        w, b = _get_hyperplane(num_feat, in_simplex, homogeneous)
        param = 'linear_separation(%d, %d, %d, %1.2f, %d)' %\
                (num_point, num_feat, num_noise, margin, int(homogeneous))
    metadata = (param, w, b)
    data = 2.0*(rand(num_feat, int((2.0/(1.0-margin))*num_point))-0.5)
    pred = array(matrix(w).T*matrix(data) + b)
    to_keep = nonzero(abs(pred) > margin)[1]
    data = data[:, to_keep]
    data = data[:, :num_point]

    # label based on w and b
    pred = array(matrix(w).T*matrix(data) + b)
    pos_idx = nonzero(pred > 0.0)[1]
    neg_idx = nonzero(pred < 0.0)[1]
    label = zeros(num_point)
    label[pos_idx] = 1.0
    label[neg_idx] = -1.0

    # generate noise dimensions
    if num_noise > 0:
        noise = 2.0*(rand(num_noise, num_point)-0.5)
        data = vstack([data, noise])

    return metadata, data, label


def linear_separation_example():
    from optwok.testtools import plot_2d
    from matplotlib.pyplot import figure

    metadata, data, label = linear_separation(200, 2, 0)
    figure()
    plot_2d(data, label)
    metadata, data, label = linear_separation(200, 2, 0, margin=0.4, homogeneous=False)
    figure()
    plot_2d(data, label)
    metadata, data, label = linear_separation(200, 1, 1, margin=0.5)
    figure()
    plot_2d(data, label)

    metadata, data, label = linear_separation(200, 2, 0, in_simplex=True)
    figure()
    plot_2d(data, label)
    metadata, data, label = linear_separation(200, 1, 1, in_simplex=True)
    figure()
    plot_2d(data, label)


#####################################################################
# Multiclass
#####################################################################

def multi_cloudgen(num_point, num_feat, num_class, edge_length):
    """Generate numclass number of Gaussian point clouds,
    centered around the simplex of numclass."""
    assert(num_feat >= num_class)
    centers = zeros((num_feat, num_class))
    centers[:num_class, :num_class] = (edge_length/sqrt(2))*eye(num_class)
    midpoint = mean(centers, axis=1)
    midpoint.shape = (num_feat, 1)
    centers = centers - kron(ones((1, num_class)), midpoint)

    if isinstance(num_point, int):
        num_point_list = []
        for ix in range(num_class):
            num_point_list.append(num_point)
        num_point = num_point_list
    metadata = 'multi_cloudgen(%s,%d,%d,%3.2f)' % (num_point, num_feat, num_class, edge_length)

    pointcloud = zeros((num_feat, 0))
    labels = array([], dtype=int)
    for ix in range(num_class):
        cur_cloud = (kron(ones((num_point[ix], 1)), centers[:, ix]).T
                     + randn(num_feat, num_point[ix]))
        pointcloud = concatenate((pointcloud, cur_cloud), axis=1)
        labels = concatenate((labels, ix*ones(num_point[ix])))
    pointcloud /= edge_length/sqrt(2)

    return metadata, pointcloud, labels


def multi_cloudgen_mix(num_point, num_feat, num_class, edge_length, mix_list):
    """Generate num_class Gaussian point clouds,
    but labels are mixed depending on mix_list.

    """
    (metadata, pointcloud, labels) = multi_cloudgen(num_point, num_feat, num_class, edge_length)
    metadata += str(mix_list)
    new_labels = zeros(labels.shape, dtype=int)
    if len(mix_list) == num_class:
        # split classes
        for cid in range(num_class):
            idx = nonzero(equal(labels, cid))[0]
            if len(mix_list[cid]) > 1:
                new_labels[idx] = randint(min(mix_list[cid]), max(mix_list[cid])+1, len(idx))
            else:
                new_labels[idx] = mix_list[cid][0]
    else:
        # merge classes
        assert(len(flatten_list(mix_list)) == num_class)
        for cid in range(len(mix_list)):
            idx = array([], dtype=int)
            for mix in mix_list[cid]:
                idx = concatenate([idx, nonzero(equal(labels, mix))[0]])
            new_labels[idx] = cid
    return metadata, pointcloud, new_labels

#####################################################################
# Simulated data for sparsity experiments
#####################################################################


def random_classification_noise(labels, frac_flip, symm_flip):
    """Generate random classification noise by flipping a proportion
    of labels randomly.
    """
    if frac_flip > 0.0:
        num_ex = len(labels)
        if symm_flip:
            flip_idx = permutation(num_ex)[:round(frac_flip*num_ex)]
            labels[flip_idx] = -1.0*labels[flip_idx]
        else:
            flip_idx = permutation(num_ex)[:round(2.0*frac_flip*num_ex)]
            labels[flip_idx] = -1.0*labels[where(labels[flip_idx] > 0.0)]


def twoballs_filename(data_dir, num_examples, num_feat, num_noise_feat, frac_flip):
    """Generate filename to save data and permutations"""
    data_filename = data_dir + '/twoballs_n=%d_%d:%d_rcn=%1.1f.csv'\
        % (num_examples, num_feat, num_noise_feat, frac_flip)
    perm_filename = data_dir + '/twoballs_n=%d_%d:%d_rcn=%1.1f_perm.txt'\
        % (num_examples, num_feat, num_noise_feat, frac_flip)
    return (data_filename, perm_filename)


def save_cloudgen(data_dir, num_noise_feat, frac_flip, symm_flip=True,
                  num_ex=100, num_feat=10, frac_pos=0.5, width=1.5, num_perms=50):
    """Generate two gaussians, with noise dimensions and
    random classification noise, and save the dataset.
    """
    assert(frac_flip >= 0.0 and frac_flip <= 0.5)
    (metadata, examples, labels) = cloudgen(num_ex, num_feat, num_noise_feat,
                                            frac_pos, width)
    random_classification_noise(labels, frac_flip, symm_flip)
    (data_filename, perm_filename) = twoballs_filename(data_dir, num_ex, num_feat, num_noise_feat,
                                                       frac_flip)
    outfile = DatasetFileCSV(data_filename, 'vec', verbose=True)
    outfile.writelines(examples, labels)

    handle = open(perm_filename, 'w')
    for iperm in range(num_perms):
        cur_mix = permutation(num_ex)
        handle.write(' '.join(map(str, cur_mix.tolist())))
        handle.write('\n')
    handle.close()


def sparse_data(data_dir):
    """Generate data to investigate sparsity,
    with respect to both features and support vectors."""
    seed(1)
    num_feat = 10
    num_examples = [50, 100]
    noise_feats = [0, 10, 100, 200]
    rcns = [0.0, 0.1, 0.2, 0.3]

    num_datasets = len(noise_feats)
    assert(len(rcns) == num_datasets)

    for num_ex in num_examples:
        for num_noise_feat in noise_feats:
            save_cloudgen(data_dir, num_noise_feat, 0.0, num_ex=num_ex, num_feat=num_feat)
        (data_filename, perm_filename) = twoballs_filename(data_dir, num_ex, num_feat, 0, 0.0)
        # Add random classification noise in the training
        for rcn in rcns[1:]:
            (df, pf) = twoballs_filename(data_dir, num_ex, num_feat, 0, rcn)
            link(data_filename, df)
            link(perm_filename, pf)
        for ix in range(1, num_datasets):
            (dfs, pfs) = twoballs_filename(data_dir, num_ex, num_feat, noise_feats[ix], 0.0)
            (df, pf) = twoballs_filename(data_dir, num_ex, num_feat, noise_feats[ix], rcns[ix])
            link(dfs, df)
            link(pfs, pf)

#####################################################################
# Gaussian Process regression
#####################################################################


def mean_func1(x, shift=0):
    from numpy import sin
    x += shift
    return 2*sin(x)+sin(3*x+1)


def mean_func2(x, shift=0):
    from numpy import cos
    x += shift
    return 2*cos(x)+cos(3*x+1)


def gp_gen(num_point, num_dim, domain, noise_level, mix_list=[[0, 1], [2]]):
    """Generate matrix variate normally distributed data"""
    reg_param = 1e-8
    num_class = len(flatten_list(mix_list))
    X = domain*rand(num_dim, num_point)
    Kx = GaussKernel(1.0)
    Kx.compute(X)
    Ky = list2matrix(mix_list, neg_corr=True)
    K = JointKernel(Kx, Ky)

    L = real(sqrtm(0.5*(K.kmat)+reg_param*eye(num_point*num_class)).T)
    mu = zeros((num_class, num_point))

    Y = L*matrix(randn(num_point*num_class, 1))
    Y.shape = (num_point, num_class)
    Y = real(Y.T)
    Y += mu + noise_level*randn(num_class, num_point)
    Y = array(Y)
    return (X, Y)


def gp_gen_pair(num_pairs, num_dim, domain, noise_level, mix_list=[[0, 1], [2]], pos_label=False):
    """Generate data from a multioutput Gaussian process,
    The outputs are irregular, and are dependent on context Z"""
    num_class = len(flatten_list(mix_list))
    num_ex = num_class*num_pairs
    num_ex = max(num_ex, num_pairs)
    (X_full, Y_full) = gp_gen(num_ex, num_dim, domain, noise_level, mix_list)
    X = zeros((X_full.shape[0], 0))
    Y = array([])
    Z = array([])
    for ix in range(num_class):
        perm = permutation(num_ex)
        X = hstack([X, X_full[:, perm[:num_pairs]]])
        Y = hstack([Y, Y_full[ix, perm[:num_pairs]]])
        Z = concatenate((Z, ix*ones(num_pairs)))
    if pos_label:
        min_y = min(Y) - 0.1
        Y -= min_y
    return (X, Z, Y)


def plot_missing_data():
    """A demo of gp_gen_pair"""
    from matplotlib.pyplot import plot, show

    colour_list = ['b', 'g', 'r', 'c', 'm', 'k']
    (X, Z, Y) = gp_gen_pair(20, 1, 5, 0.1)
    for zix in range(3):
        cur_idx = flatnonzero(Z == zix)
        plot(X[:, cur_idx].flatten(), Y[:, cur_idx].flatten(),
             colour_list[zix]+'o', hold=True)
    show()


def save_gp_data(x, y, dataname, data_dir='.'):
    """Save Gaussian process data"""
    dataset = DatasetBase()
    dataset.name = dataname
    dataset.filename = dataset.name + '.pkl'
    dataset.examples = x
    dataset.labels = y
    (dataset.num_features, dataset.num_examples) = x.shape
    dataset.num_class = y.shape[0]
    dataset.perm_filename = ''
    dataset.perms = zeros((50, dataset.num_examples), dtype=int)
    for ix in range(50):
        dataset.perms[ix] = permutation(dataset.num_examples)
    dataset.perm_idx = -1
    dataset.data_dir = data_dir
    dataset.frac_train = 0.7

    save('%s/%s' % (data_dir, dataset.filename), dataset)


def save_gp_context_data(x, z, y, dataname, data_dir='.'):
    """Save Gaussian process with context (z) data"""
    dataset = DatasetBase()
    dataset.name = dataname
    dataset.filename = dataset.name + '.pkl'
    dataset.examples = x
    dataset.context = z
    dataset.labels = y
    (dataset.num_features, dataset.num_examples) = x.shape
    dataset.num_class = y.shape[0]
    dataset.perm_filename = ''
    dataset.perms = zeros((50, dataset.num_examples), dtype=int)
    for ix in range(50):
        dataset.perms[ix] = permutation(dataset.num_examples)
    dataset.perm_idx = -1
    dataset.data_dir = data_dir
    dataset.frac_train = 0.7

    save('%s/%s' % (data_dir, dataset.filename), dataset)


def gp_example():
    """Plot some points"""
    from matplotlib.pyplot import plot, figure

    num_ex = 200
    domain = 5
    noise_level = 0.3
    (X, Y) = gp_gen(num_ex, 1, domain, noise_level)
    dataname = 'gaussproc_03_%d_%1.1f' % (num_ex, noise_level)
    save_gp_data(X, Y, dataname, data_dir='/local/cong/Data/muloutreg')

    data = load('Data/muloutreg/gaussproc_03_%d_%1.1f.pkl' % (num_ex, noise_level))
    figure()
    plot(data.examples.T, data.labels.T, '+', hold=True)

#####################################################################
# learning with teacher
#####################################################################


def teacher_data(num_train=100, num_test=300, num_feat=5, num_feat_teach=10,
                 width=1.0):
    """Generate two Gaussians and then split features
    into normal features and teacher features
    """
    (metadata, examples, labels) = cloudgen(num_train+num_test, num_feat+num_feat_teach,
                                            0, 0.5, width)
    cur_mix = permutation(num_train+num_test)
    examples = examples[:, cur_mix]
    labels = labels[cur_mix]
    train_ex = examples[:num_feat, :num_train]
    teach_ex = examples[-num_feat_teach:, :num_train]
    pred_ex = examples[:num_feat, -num_test:]
    train_lab = labels[:num_train]
    pred_lab = labels[-num_test:]

    return (train_ex, teach_ex, train_lab, pred_ex, pred_lab)


if __name__ == '__main__':
    data = teacher_data()
    (train_ex, teach_ex, train_lab, test_ex, test_lab) = data
    print(train_ex.shape)
    print(teach_ex.shape)
    print(train_lab.shape)
    print(test_ex.shape)
    print(test_lab.shape)
