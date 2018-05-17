"""
Spherical k-means
- Adapted from https://github.com/justinsalamon/skm
"""

from datetime import datetime
import json
import pickle

import numpy as np
import sklearn
from sklearn.decomposition import PCA
import yaml


class SKM(object):
    """
    Spherical k-means with PCA whitening
    - Based on: Coats & Ng, "Learning Feature Representations with K-means", 2012
    """

    _ARGS = [
        'k',
        'variance_explained',
        'max_epochs',
        'assignment_change_eps',
        'standardize',
        'normalize',
        'pca_whiten',
        'do_pca',
        'verbose',
    ]

    _PARAMS = _ARGS + [
        'epoch',
        'assignment_change',
        'nfeatures',
        'nsamples',
        'D',
        'assignment',
        'prev_assignment',
        'initialized',
        'mus',
        'sigmas',
    ]

    # Based on sklearn 0.15.2
    _PCA_ARGS = [
        'n_components',
        'copy',
        'whiten',
    ]

    _PCA_PARAMS = _PCA_ARGS + [
        'components_',
        'explained_variance_',
        'explained_variance_ratio_',
        'mean_',
        'n_samples_',
        'noise_variance_',
    ]

    @property
    def args(self):
        return {k: getattr(self, k, None) for k in SKM._ARGS}

    @property
    def params(self):
        return {k: getattr(self, k, None) for k in SKM._PARAMS}

    @property
    def pca_args(self):
        return {k: getattr(self.pca, k, None) for k in SKM._PCA_ARGS}

    @property
    def pca_params(self):
        return {k: getattr(self.pca, k, None) for k in SKM._PCA_PARAMS}

    def __init__(
        self,
        k=500,
        variance_explained=0.99,
        max_epochs=100,
        assignment_change_eps=0.01,
        standardize=False,
        normalize=False,
        pca_whiten=True,
        do_pca=True,
        verbose=True,
    ):
        # Args
        self.k = k
        self.variance_explained = variance_explained
        self.max_epochs = max_epochs
        self.assignment_change_eps = assignment_change_eps
        self.standardize = standardize
        self.normalize = normalize
        self.pca_whiten = pca_whiten
        self.do_pca = do_pca
        self.verbose = verbose

        # Params
        self.epoch = 0
        self.assignment_change = np.inf
        self.nfeatures = None
        self.nsamples = None
        self.D = None # centroid dictionary
        self.assignment = None # assignment vector
        self.prev_assignment = None # previous assignment vector
        self.initialized = False
        self.mus = None
        self.sigmas = None

        # pca
        self.pca = PCA(n_components=self.variance_explained, copy=False, whiten=self.pca_whiten)

    def _pca_fit_transform(self, X):
        '''PCA fit and transform the data'''
        self._log('_pca_fit_transform')
        data = self.pca.fit_transform(X.T) # transpose for PCA
        return data.T # transpose back

    def _pca_fit(self, X):
        '''PCA fit only (don't transform the data)'''
        self._log('_pca_fit')
        self.pca.fit(X.T)

    def _pca_transform(self, X):
        '''PCA transform only (must call fit or fit_transform first)'''
        self._log('_pca_transform')
        data = self.pca.transform(X.T)
        return data.T

    def _normalize_samples(self, X):
        '''Normalize the features of each sample so that their values sum to one (might not make sense for all data)'''
        self._log('_normalize_samples')
        data = sklearn.preprocessing.normalize(X, axis=0, norm='l1')
        return data

    def _standardize_fit(self, X):
        '''Compute mean and variance (of each feature) for standardization'''
        self._log('_standardize_fit')
        self.mus = np.mean(X, 1)
        self.sigmas = np.std(X, 1)

    def _standardize_transform(self, X):
        '''Standardize input data (assumes standardize_fit already called)'''
        self._log('_standardize_transform')
        data = X.T - self.mus
        data /= self.sigmas
        return data.T

    def _standardize_fit_transform(self, X):
        '''Compute means and variances (of each feature) for standardization and standardize'''
        self._standardize_fit(X)
        data = self._standardize_transform(X)
        return data

    def _init_centroids(self):
        '''Initialize centroids randomly from a normal distribution and normalize (must call _set_dimensions first)'''
        # Sample randomly from normal distribution
        self.D = np.random.normal(size=[self.nfeatures, self.k])
        self._normalize_centroids()
        self.initialized = True

    def _normalize_centroids(self):
        '''Normalize centroids to unit length (using l2 norm)'''
        self.D = sklearn.preprocessing.normalize(self.D, axis=0, norm='l2')

    def _update_centroids(self, X):
        '''Update centroids based on provided sample data X'''
        S = np.dot(self.D.T, X)
        # centroid_index = np.argmax(S, 0)
        centroid_index = S.argmax(0) # slightly faster
        s_ij = S[centroid_index, np.arange(self.nsamples)]
        S = np.zeros([self.k, self.nsamples])
        S[centroid_index, np.arange(self.nsamples)] = s_ij
        self.D += np.dot(X, S.T)
        self.prev_assignment = self.assignment
        self.assignment = centroid_index

    def _update_centroids_memsafe(self, X):
        '''Update centroids based on provided sample data X. Try to minimize memory usage.'''
        Dt = self.D.T
        centroid_index = np.zeros(X.shape[1], dtype='int')
        s_ij = np.zeros(X.shape[1])
        for n,x in enumerate(X.T):
            dotprod = np.dot(Dt, x)
            centroid_index[n] = np.argmax(dotprod)
            s_ij[n] = dotprod[centroid_index[n]]

        # S = np.zeros([self.k, self.nsamples])
        # S[centroid_index, np.arange(self.nsamples)] = s_ij
        # self.D += np.dot(X, S.T)
        for n in np.arange(self.k):
            s = np.zeros(X.shape[1])
            s[centroid_index==n] = s_ij[centroid_index==n]
            self.D[:,n] += np.dot(X, s)

        self.prev_assignment = self.assignment
        self.assignment = centroid_index

    def _update_centroids_memsafe_fast(self, X):
        '''Update centroids based on provided sample data X. Try to minimize memory usage. Use weave for efficiency.'''
        Dt = self.D.T
        centroid_index = np.zeros(X.shape[1], dtype='int')
        s_ij = np.zeros(X.shape[1])
        for n,x in enumerate(X.T):
            dotprod = np.dot(Dt, x)
            centroid_index[n] = np.argmax(dotprod)
            s_ij[n] = dotprod[centroid_index[n]]

        # S = np.zeros([self.k, self.nsamples])
        # S[centroid_index, np.arange(self.nsamples)] = s_ij
        # self.D += np.dot(X, S.T)
        S = np.zeros([self.nsamples, self.k])
        S[np.arange(self.nsamples), centroid_index] = s_ij
        self.D += np.dot(X, S)

        # for n in np.arange(self.k):
        #     s = np.zeros(X.shape[1])
        #     s[centroid_index==n] = s_ij[centroid_index==n]
        #     self.D[:,n] += np.dot(X, s)

        # nfeatures = X.shape[0]
        # nsamples = X.shape[1]
        # s = np.zeros(nsamples)
        # k = self.k
        # D = self.D
        # dotproduct_command = r"""
        # for (int n=0; n<k; n++)
        # {
        #     for (int m=0; m<nsamples; m++)
        #     {
        #        if (centroid_index[m]==n)
        #             s[m] = s_ij[m];
        #         else
        #             s[m] = 0;
        #     }
        #
        #     for (int f=0; f<nfeatures; f++)
        #     {
        #         float sum = 0;
        #         for (int i=0; i<nsamples; i++)
        #         {
        #             sum += X[f*nsamples + i] * s[i];
        #         }
        #         D[f*k + n] += sum;
        #     }
        # }
        # """
        # scipy.weave.inline(dotproduct_command, ['k','nsamples','centroid_index','s','s_ij','nfeatures','X','D'])

        self.prev_assignment = self.assignment
        self.assignment = centroid_index

    def _update_centroids_cuda(self, X):
        '''Update centroids based on provided sample data X using GPU via cuda'''
        # S = np.dot(self.D.T, X)
        Xcuda = cm.CUDAMatrix(X)
        S = cm.dot(cm.CUDAMatrix(self.D).T, Xcuda)
        # centroid_index = S.argmax(0) # slightly faster
        centroid_index = S.asarray().argmax(axis=0)
        # s_ij = S[centroid_index, np.arange(self.nsamples)]
        s_ij = S.asarray()[centroid_index, np.arange(self.nsamples)]
        S = np.zeros([self.nsamples, self.k])
        # S[centroid_index, np.arange(self.nsamples)] = s_ij
        S[np.arange(self.nsamples), centroid_index] = s_ij
        self.D += cm.dot(Xcuda, cm.CUDAMatrix(S)).asarray()
        self.prev_assignment = self.assignment
        self.assignment = centroid_index

    def _init_assignment(self):
        '''Initialize assignment of samples to centroids (must call _set_dimensions first)'''
        self.prev_assignment = np.zeros(self.nsamples) - 1
        self.assignment = None

    def _set_dimensions(self, X):
        '''Set dimensions (number of features, number of samples) based on dimensions of input data X'''
        self.nfeatures, self.nsamples = X.shape

    def _compute_assignment_change(self):
        '''Compute the fraction of assignments changed by the latest centroid update (value between 0 to 1)'''
        self.assignment_change = np.mean(self.assignment != self.prev_assignment)

    def fit(self, X, memsafe=False, cuda=False):
        '''Fit k centroids to input data X until convergence or max number of epochs reached.'''
        self._log('fit')

        # Normalize data (per sample)
        if self.normalize:
            X = self._normalize_samples(X)

        # Standardize data (across samples)
        if self.standardize:
            X = self._standardize_fit_transform(X)

        # PCA fit and whiten the data
        if self.do_pca:
            X = self._pca_fit_transform(X)

        # Store dimensions of whitened data
        self._set_dimensions(X)

        # Initialize centroid dictionary
        self._init_centroids()

        # Initialize assignment
        self._init_assignment()

        # Iteratively update and normalize centroids
        self._log('fit: iterating...')
        while True:
            if self.epoch >= self.max_epochs:
                self._log(f'fit: done: epoch[{self.epoch}] < max_epochs[{self.max_epochs}]')
                break
            if self.assignment_change <= self.assignment_change_eps:
                self._log(
                    f'fit: done: assignment_change[{self.assignment_change}] < '
                    f'assignment_change_eps[{self.assignment_change_eps}]',
                )
                break
            if memsafe:
                self._update_centroids_memsafe_fast(X)
            elif cuda:
                self._update_centroids_cuda(X)
            else:
                self._update_centroids(X)
            self._normalize_centroids()
            self._compute_assignment_change()
            self.epoch += 1
            self._report_status()

    def fit_minibatch(self, X):
        '''
        Fit k centroids to input data X until convergence or max number of epochs reached. Assumes X is a mini-batch
        from a larger sample set. The first batch is used to initialize the algorithm (dimensions).
        '''
        self._log('fit_minibatch')

        # Normalize data (per sample)
        if self.normalize:
            X = self._normalize_samples(X)

        # If this is the first batch, use it to initialize
        if not self.initialized:
            self._log('fit_minibatch: initializing')
            # Standardize data
            if self.standardize:
                X = self._standardize_fit_transform(X)
            # PCA whiten the data
            if self.do_pca:
                X = self._pca_fit_transform(X)
            # Store dimensions of whitened data
            self._set_dimensions(X)
            # Initialize centroid dictionary
            self._init_centroids()
            # Initialize assignment
            self._init_assignment()
        else:
            self._log('fit_minibatch: already initialized')
            if self.standardize:
                X = self._standardize_transform(X)
            if self.do_pca:
                X = self._pca_transform(X)
            # Reset epochs and assignments
            self.epoch = 0
            self._init_assignment()
            self.assignment_change = np.inf

        # Iteratively update and normalize centroids
        self._log('fit_minibatch: iterating...')
        while True:
            if self.epoch >= self.max_epochs:
                self._log(f'fit: done: epoch[{self.epoch}] < max_epochs[{self.max_epochs}]')
                break
            if self.assignment_change <= self.assignment_change_eps:
                self._log(
                    f'fit: done: assignment_change[{self.assignment_change}] < '
                    f'assignment_change_eps[{self.assignment_change_eps}]',
                )
                break
            self._update_centroids(X)
            self._normalize_centroids()
            self._compute_assignment_change()
            self.epoch += 1
            self._report_status()

    def _report_status(self):
        self._log(f'epoch[{self.epoch}] assignment_change[{self.assignment_change}]')

    def transform(self, X, rectify=False, nHot=0):
        '''Transform samples X (each column is a feature vector) to learned feature space'''
        self._log('transform')

        # print("DEBUG: entered skm.transform")
        # Normalize data (per sample)
        if self.normalize:
            X = self._normalize_samples(X)

        # Standardize data (across samples)
        if self.standardize:
            X = self._standardize_fit_transform(X)

        # print("DEBUG: skipped normalized/standardize")

        # PCA whiten
        if self.do_pca:
            X = self._pca_transform(X)
        # X = np.random.rand(149, 173)
        # print("DEBUG: did PCA")

        # Dot product with learned dictionary
        X = np.dot(X.T, self.D)
        # print("DEBUG: did dot product")

        if rectify:
            X = np.maximum(X, 0)

        # x-hot coding instead of just dot product
        if nHot > 0:
            indices = np.argsort(X)
            for n,x in enumerate(X):
                x[indices[n][0:-nHot]] = 0
                x[indices[n][-nHot:]] = 1
        return X.T

    def _log(self, event, **kwargs):
        """Simple, ad-hoc logging"""
        event = f'[{type(self).__name__}] {event}'
        if self.verbose:
            t = datetime.utcnow().isoformat()
            t = t[:23]  # Trim micros, keep millis
            t = t.split('T')[-1]  # Trim date for now, since we're primarily interactive usage
            # Display timestamp + event on first line
            print('[%s] %s' % (t, event))
            # Display each (k,v) pair on its own line, indented
            for k, v in kwargs.items():
                v_yaml = yaml.safe_dump(json.loads(json.dumps(v)), default_flow_style=True, width=1e9)
                v_yaml = v_yaml.split('\n')[0]  # Handle documents ([1] -> '[1]\n') and scalars (1 -> '1\n...\n')
                print('  %s: %s' % (k, v_yaml))


def spherical_kmeans_viz(X_raw, k):
    '''
    Given data input X (each column is a datapoint, number of rows if dimensionality of the data), and desired number of
    means k, compute the means (dictionary D) and cluster assignment S using the spherical k-means algorithm (cf. Coats
    & NG, 2012)

    Data is assumed to be normalized.
    '''

    # Step 1: whiten inputs using PCA
    n_components = 0.99 # explain 99% of the variance
    pca = PCA(n_components=n_components, copy=True, whiten=True)
    X = pca.fit_transform(X_raw.T)
    X = X.T
    # X = X_raw

    dim, N = X.shape
    # print X.shape

    # Step 2: k-means
    # Step 2.1: initialize dictionary D
    D = np.random.normal(size=[dim,k]) # sample randomly from normal distribution
    D = sklearn.preprocessing.normalize(D, axis=0, norm='l2') # normalize centroids
    # print 'D.shape', D.shape

    # Step 2.2: initialize code vectors (matrix) S

    # Step 2.2: update until convergence
    epoc = 0
    max_epocs = 100
    change = np.inf
    change_eps = 0.001
    prev_index = np.zeros(N)
    while epoc < max_epocs and change > change_eps:
        S = np.dot(D.T,X)
        # print 'S.shape', S.shape
        centroid_index = np.argmax((S), 0) # np.abs? NO!
        # print 'centroid_index.shape', centroid_index.shape
        s_ij = S[centroid_index, np.arange(N)] # dot products already calculated
        S = np.zeros([k,N])
        S[centroid_index, np.arange(N)] = s_ij
        D += np.dot(X,S.T)
        D = sklearn.preprocessing.normalize(D, axis=0, norm='l2') # normalize

        # Compute change
        change = np.mean(centroid_index != prev_index)
        prev_index = centroid_index
        epoc += 1
        print("EPOC: {:d} CHANGE: {:.4f}".format(epoc,change))

        # # Visualize clustering
        # pl.figure()
        # colors = ['b','r','g', 'c', 'm', 'y', 'k']
        # for n,point in enumerate(X.T):
        #     pl.plot(point[0],point[1], colors[centroid_index[n]] + 'o')
        # pl.axhline(0, color='black')
        # pl.axvline(0, color='black')
        # for n,centroid in enumerate(D.T):
        #     # centroid = pca.inverse_transform(centroid)
        #     pl.plot(centroid[0],centroid[1],colors[n] + 'x')
        # pl.show()
