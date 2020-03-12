import numpy as np
import scipy.stats as st

from skpro.distributions.distribution_base import Mode
from skpro.distributions.covariance import CovarianceMatrix
from skpro.distributions.distribution_normal import Normal
from skpro.distributions.distribution_multivariate_normal import MultiVariateNormal
from skpro.distributions.distribution_mixture import Mixture
from skpro.distributions.distribution_custom import CustomDistribution

from skpro.distributions.component.support import Support
from skpro.distributions.component.set import BaseSet



def test_univariate_parameterization():

    # test set/get_parameters inheritence on standard normal
    a = Normal()
    parameters = a.get_params()
 
    assert(a.variateSize() == 1)
    assert(a.vectorSize() == 1)
    assert(parameters['loc'] == 0.0)
    assert(parameters['scale'] == 1.0)
    
    #test equality override
    assert(a == Normal(loc=0, scale=1))
    
    a.set_params(loc = 3.0)
    assert(a.get_params()['loc'] == 3.0)

    #test vectorization of univariate
    b = Normal([5,10, 0], [20, 30, 10])
    
    assert(b.variateSize() == 1)
    assert(b.vectorSize() == 3)
    assert(b.get_params() == {'loc': [5, 10, 0], 'scale': [20, 30, 10]})

    assert(b.get_params(0) == {'loc': 5, 'scale': 20})
    assert(b.get_params(1) == {'loc': 10, 'scale': 30})


def test_subsetting():
    
    n = Normal([0, 0, 0], [1, 1.5, 0.5])
    assert(n[0] == Normal())
    assert(n[2] == Normal(0, 0.5))
    assert(len(n[1:3]) == 2)
    assert(n[1:3] == Normal([0, 0], [1.5,0.5]))
    assert(n[0:3:2] == Normal(loc=[0, 0], scale=[1, 0.5]))


def test_multivariate_parameterization():
    
    #test instanciation of multivariate normal
    cov = [[1, 0],[0, 1]]
    c = MultiVariateNormal([5,10], cov)
    cmat = CovarianceMatrix(cov)

    assert(c.variateSize() == 2)
    assert(c.vectorSize() == 1)
    assert (c.get_params(deep = False) == {'loc': [5, 10], 'cov': cmat})
    assert (c.get_params(deep = True) == {'loc': [5, 10], 'cov': cmat, 'cov__data': cmat.cov()})

    #test vectorization of multivariate normal
    cov1 = [[1, 0],[0, 1]]
    cov2 = [[1, 0.5],[0.5, 1]]
    c = MultiVariateNormal(loc = [[5,10], [0,0]], cov = [cov1 , cov2])
    
    assert(c.variateSize()== 2)
    assert(c.vectorSize() == 2)
    assert (c.get_params(0, deep = False) == {'loc': [5, 10], 'cov': CovarianceMatrix(cov1)})
    assert (c.get_params(1, deep = False) == {'loc': [0, 0], 'cov': CovarianceMatrix(cov2)})


def test_univariate_function_dpqr():
    
    rtol = 1e-05
    atol = 1e-08

    #test point pdf/cdf on vectorized univariate
    a = Normal([0, 0.1, 0.2], [1, 1, 1.2])

    assert(np.allclose(a.pdf(0), [0.3989422804014327, 0.3969525474770118, 0.3278664300849499], rtol, atol))
    assert(np.allclose(a.cdf(0), [0.5, 0.46017216, 0.43381617], rtol, atol))

    #test batch_wise pdf/cdf on vectorized univariate
    b = Normal([0,0.1], [1, 1.2])

    assert (np.allclose(b.pdf([0, 1.0]), [[0.3989422804014327, 0.331299555215285], [0.24197072451914337, 0.2509478601290037]], rtol, atol))
    assert (np.allclose(b.cdf([0, 1.0]), [[0.5, 0.46679325], [0.84134475, 0.77337265]], rtol, atol))
    
    #test batch_wise pdf/cdf on vectorized univariate
    d = Normal([0, 0.1], [1, 1.2])
    d.setMode(Mode.ELEMENT_WISE)

    assert (np.allclose(d.pdf([0, 1.0, 0]),  [0.3989422804014327, 0.2509478601290037, 0.3989422804014327], rtol, atol))
    assert (np.allclose(d.cdf([0, 1.0, 0]), [0.5, 0.7733726476231317, 0.5], rtol, atol))
    
    #squared_norm (ie. target free function call)
    assert (np.allclose(a.squared_norm(), [0.28209479177387814, 0.28209479177387814, 0.24557777988165252], rtol, atol))


def test_multivariate_function_dpqr():
    
    rtol = 1e-05
    atol = 1e-08
    
    cov1 = [[15.8,9.6,-12],[9.6,21.7,-17.25], [-12,-17.25,18.5]]
    cov2 = [[8.5, 37/4 ,-25/4],[37/4, 227/10, -219/20], [-25/4, -219/20, 87/10]]
    cov = [cov1, cov2]

    loc = [[0,2, 0], [0,1,3]]
    n = MultiVariateNormal(loc, cov)
    
    x = [[0, 1, 3], [0, 2, 4]]
    
    #test point pdf
    out = n.pdf(x[0])
    assert(np.allclose(out, [0.00089442, 0.00370004], rtol, atol))
 
    #test batch_wise pdf
    out = n.pdf(x)
    assert (np.allclose(out[0][0], st.multivariate_normal(np.array(loc[0]), np.array(cov[0])).pdf(x[0]), rtol, atol))
    assert (np.allclose(out[0][1], st.multivariate_normal(np.array(loc[1]), np.array(cov[1])).pdf(x[0]), rtol, atol))
    assert (np.allclose(out[1][0], st.multivariate_normal(np.array(loc[0]), np.array(cov[0])).pdf(x[1]), rtol, atol))
    assert (np.allclose(out[1][1], st.multivariate_normal(np.array(loc[1]), np.array(cov[1])).pdf(x[1]), rtol, atol))
    
    #test element wise pdf
    n.setMode(Mode.ELEMENT_WISE)
    out = n.pdf(x)

    assert (np.allclose(out[0], st.multivariate_normal(np.array(loc[0]), np.array(cov[0])).pdf(x[0]), rtol, atol))
    assert (np.allclose(out[1], st.multivariate_normal(np.array(loc[1]), np.array(cov[1])).pdf(x[1]), rtol, atol))


def test_mixture_distribution():
    
    n = [Normal(0, 1), Normal(0, 2)]
    m1 = Mixture(n, [0.5, 0.5])
    
    #test list of distribution eq
    assert(n == [Normal(loc=0, scale=1), Normal(loc=0, scale=2)])
    
    #test instanciation of mixture
    assert(m1.vectorSize() == 2)
    assert(m1.weights == [[0.5, 0.5], [0.5, 0.5]])
    
    #test get_params accessor
    dic1 = {'distribution': n,  'weights': [[0.5, 0.5], [0.5, 0.5]]}

    dic2 = {'distribution': n, 'weights': [[0.5, 0.5], [0.5, 0.5]],
           'distribution_0_loc': 0.0, 
           'distribution_0_scale': 1.0, 
           'distribution_1_loc': 0.0, 
           'distribution_1_scale': 2.0}

    dic3 = {'distribution': n[0], 'weights': [0.5, 0.5],
           'distribution__loc': 0.0, 
           'distribution__scale': 1.0}
    
    assert(m1.get_params(deep = False) == dic1)
    assert(m1.get_params() == dic2)
    assert(m1.get_params(0) == dic3)
    
    #test vectorized univariate pdf
    n1 = Normal([0,0], [1,2])
    n2 = Normal([0,0], [1,3])
    m2 =  Mixture([n1, n2],  [[0.5, 0.5],[0.4,0.6]])
    X = [0, 0.2, 0.4, 0.3]
    pdf = m2.pdf(X)
    
    assert(m2.vectorSize() == 2)
    assert(m2.variateSize() == 1)

    assert(pdf[0][0] == n1[0].pdf(X[0])*0.5 + n1[1].pdf(X[0])*0.5)
    assert(pdf[0][1] == n2[0].pdf(X[0])*0.4 + n2[1].pdf(X[0])*0.6)
    
    assert(pdf[1][0] == n1[0].pdf(X[1])*0.5 + n1[1].pdf(X[1])*0.5)
    assert(pdf[1][1] == n2[0].pdf(X[1])*0.4 + n2[1].pdf(X[1])*0.6)
    
    assert(pdf[2][0] == n1[0].pdf(X[2])*0.5 + n1[1].pdf(X[2])*0.5)
    assert(pdf[2][1] == n2[0].pdf(X[2])*0.4 + n2[1].pdf(X[2])*0.6)
    


def test_custom_distribution():
    
    func = lambda x, sup, inf : max(min(x-inf, sup-inf),0)/(sup-inf)

    u_set = BaseSet(sup = 1, inf = 0)
    support = Support(u_set)

    n = CustomDistribution('', cdf_func = func, support = support, sup = 1, inf = 0)
    
    assert (n.cdf(0) == 0)
    assert (n.cdf(5) == 1)
    assert (n.cdf(0.25) == 0.25)



if __name__ == "__main__":
     test_univariate_parameterization()
     test_subsetting()
     test_multivariate_parameterization()
     test_univariate_function_dpqr()
     test_multivariate_function_dpqr()
     test_mixture_distribution()
     test_custom_distribution()

     
     

    