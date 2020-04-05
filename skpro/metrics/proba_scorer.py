
import numpy as np 
from skpro.distributions.distribution_base import Mode
from skpro.estimators.base import ProbabilisticEstimator
from skpro.metrics.proba_loss_base import ProbabilisticLossFunction

class ProbabilisticScorer :
    """Scorer class used to return the average loss of an estimator (given a training data set)
    
     Parameters
     ----------
     loss :  skpro probabilistic loss object
           loss function object used by the scorer
           raise an error if not a skpro probabilistic loss object
     """

    def __init__(self, loss):
        
        if not isinstance(loss, ProbabilisticLossFunction):
            raise ValueError("loss arg is not a probabilistic loss")
            
        self.loss_func = loss


    def __call__(self, estimator, X, y, mode = 'mean', return_scores = False):
        """ () operator override that returns the total loss of the estimator according to the selected mode
        
           Parameters
           ----------
           estimator :  skpro probabilistic estimator object 
                     (else raise an error) (see skpro.estimators.base)
           
           X : array-like, shape = (n_samples, m_features)
            Test samples
           
           y : array-like, shape = (n_samples)
            Test targets
            
           mode : string
                 specify the mode of output : 1. 'mean' return the mean loss [by default]
                 2. 'total' return the total loss
                     
           return_scores : boolean
                 specify wether the vector of losses should be added as second output 
                 
                 
            Return
            -----------
            
            out : scalar of type float
                   contain the total agregated loss (averaged or not depending on the slected mode)
                   
            score : array-like
                 contains the vector of non aggregated losses (only return if 'retrun_scores' is set to TRUE)

        """
        
        if not isinstance(estimator, ProbabilisticEstimator):
            raise ValueError("estimator is not a probabilistic estimator")
        
        if not mode in ['mean', 'total']:
            mode = 'mean'

        y_proba = estimator.predict_proba(X)
        y_proba.setMode(Mode.ELEMENT_WISE)
        
        score = self.loss_func(y_proba, y)
        
        if(mode == 'average'):
            out = np.mean(score)
        else :
            out = np.sum(score)

        if return_scores :
            return out, score
        else :
            return out

