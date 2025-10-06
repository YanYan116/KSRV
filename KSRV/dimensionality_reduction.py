import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import umap
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, NMF, SparsePCA, KernelPCA
from sklearn.cross_decomposition import PLSRegression


def process_dim_reduction(method='pca', n_dim=10):
    """
    Default linear dimensionality reduction method. For each method, return a
    BaseEstimator instance corresponding to the method given as input.          
	Attributes
    -------
    method: str, default to 'pca'
    	Method used for dimensionality reduction.
    	Implemented: 'pca', 'ica', 'fa' (Factor Analysis), 
    	'nmf' (Non-negative matrix factorisation), 'sparsepca' (Sparse PCA).
    
    n_dim: int, default to 10
    	Number of domain-specific factors to compute.
    Return values
    -------
    Classifier, i.e. BaseEstimator instance
    """

    if method.lower() == 'pca':
        clf = PCA(n_components=n_dim)

    elif method.lower() == 'ica':
        print('ICA')
        clf = FastICA(n_components=n_dim)

    elif method.lower() == 'fa':
        clf = FactorAnalysis(n_components=n_dim)

    elif method.lower() == 'nmf':
        clf = MYNMF(n_components=n_dim)
        # clf = NMF(n_components=n_dim)

    elif method.lower() == 'sparsepca':
        clf = SparsePCA(n_components=n_dim, alpha=10., tol=1e-4, verbose=10, n_jobs=1)

    elif method.lower() == 'pls':
        clf = PLS(n_components=n_dim)

    elif method.lower() == 'kernelpca':
        clf = KPCA(n_components=n_dim, kernel='rbf')
        
    elif method.lower() == 'umap':
        clf = umap.UMAP(n_components=n_dim)
    
		
    else:
        raise NameError('%s is not an implemented method'%(method))

    return clf


class PLS():
    """
    Implement PLS to make it compliant with the other dimensionality
    reduction methodology.  
    (Simple class rewritting).
    """
    def __init__(self, n_components=10):
        self.clf = PLSRegression(n_components)

    def get_components_(self):
        return self.clf.x_weights_.transpose()

    def set_components_(self, x):
        pass

    components_ = property(get_components_, set_components_)

    def fit(self, X, y=None):
        self.clf.fit(X)
        return self

    def transform(self, X):
        return self.clf.transform(X)

    def predict(self, X):
        return self.clf.predict(X)
    

class KPCA():
    """
    """
    def __init__(self, n_components=10, kernel='rbf'):
        self.clf = KernelPCA(n_components,kernel=kernel)

    def get_components_(self):
        return self.clf.eigenvectors_.T

    def set_components_(self, x):
        pass

    components_ = property(get_components_, set_components_)


    def fit(self, X, y):
        self.clf.fit(X.T)
        return self

    def transform(self, X):
        return self.clf.transform(X.T)

    # def predict(self, X):
    #     return self.clf.predict(X)

class MYNMF():
    """
    """
    def __init__(self, n_components=10):
        self.clf = NMF(n_components)

    def get_components_(self):
        return self.clf.components_

    def set_components_(self, x):
        pass

    components_ = property(get_components_, set_components_)


    def fit(self, X):
        if np.min(X) < 0:
            X = X - np.min(X)  
        self.clf.fit(X)
        return self

    def transform(self, X):
        if np.min(X) < 0:
            X = X - np.min(X)  
        return self.clf.transform(X)

