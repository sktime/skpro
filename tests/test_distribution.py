from skpro.distributions.marginal_distribution import NormalDistribution
from skpro.distributions.multivariate_distribution import MultiVariateNormal

def test_univariate_parametrazation():

    # test set/get_parameters inheritence on standard normal
    a = NormalDistribution()
    parameters = a.get_params()
 
    assert(a.dimension_ == 1)
    assert(a.size_ == 1)
    assert(parameters['loc'] == 0.0)
    assert(parameters['scale'] == 1.0)
    
    a.set_params(loc = 3.0)
    assert(a.get_params()['loc'] == 3.0)

    #test vectorization of univariate
    b = NormalDistribution([5,10, 0], [20, 30, 10])
    
    assert(b.dimension_ == 1)
    assert(b.size_ == 3)
    assert(b.get_params() == {'loc': [5, 10, 0], 'scale': [20, 30, 10]})
    assert(b.get_params(0) == {'loc': 5, 'scale': 20})
    assert(b.get_params(1) == {'loc': 10, 'scale': 30})
    
    
    
def test_multivariate_parametrazation():
    
    #test instanciation of multivariate normal
    cov = [[1, 0],[0, 1]]
    c = MultiVariateNormal([5,10], cov)
    
    assert(c.dimension_ == 2)
    assert(c.size_ == 1)
    assert (c.get_params() == {'loc': [5, 10], 'cov': cov})
    
    #test vectorization of multivariate normal
    cov1 = [[1, 0],[0, 1]]
    cov2 = [[1, 0.5],[0.5, 1]]
    c = MultiVariateNormal(loc = [[5,10], [0,0]], cov = [cov1, cov2])
    
    assert(c.dimension_ == 2)
    assert(c.size_ == 2)
    assert (c.get_params(0) == {'loc': [5, 10], 'cov': cov1})
    assert (c.get_params(1) == {'loc': [0, 0], 'cov': cov2})
    
    
    
def test_univariate_function_output_mode():

    #test point pdf/cdf on vectorized univariate
    a = NormalDistribution([0, 0.1, 0.2], [1, 1, 1.2])

    assert (a.pdf(0) == [0.3989422804014327, 0.3969525474770118, 0.3581633978016395])
    assert (a.cdf(0) == [0.5, 0.460172162722971, 0.42756607029235294])
    
    #test element_wise pdf/cdf on vectorized univariate
    b = NormalDistribution([0, 0.1], [1, 1.2])

    assert (b.pdf([0, 0.25, 0.5, 0.75, 1.0], mode = 'element_wise') == [0.3989422804014327, 0.36078455058326087, 0.3520653267642995, 0.30539752869160364, 0.24197072451914337])
    assert (b.cdf([0, 0.25, 0.5, 0.75, 1.0], mode = 'element_wise') == [0.5, 0.5544571898913917, 0.691462461274013, 0.7235319158751, 0.8413447460685428])
    
    #test batch_wise pdf/cdf on vectorized univariate
    assert (b.pdf([0, 1.0], mode = 'batch_wise') == [[0.3989422804014327, 0.24197072451914337], [0.3626685387445164, 0.2598633633704894]])
    assert (b.cdf([0, 1.0], mode = 'batch_wise') == [[0.5, 0.8413447460685428], [0.46363223676261606, 0.7943431041118706]])
    
    #squared_norm (ie. target free function call)
    assert (a.squared_norm() == [0.28209479177387814, 0.28209479177387814, 0.24557777988165252])
    


if __name__ == "__main__":
     test_univariate_parametrazation()
     test_multivariate_parametrazation()
     test_univariate_function_output_mode()
    
    