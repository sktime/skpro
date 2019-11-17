import numpy as np
import math as m
import abc
    
class Distribution(metaclass=abc.ABCMeta):
    pass
        
class NormalDistribution(Distribution) :
    
     def __init__(self, loc = 0.0, scale =1.0):
        self.loc = loc 
        self.scale = scale
        
     def name(self):
         return "normal"
     
     def mean(self):
        return self.loc
         
     def pdf(self, y):
        return np.exp(-(y - self.loc)**2/(2*self.scale))/np.sqrt(2*np.pi*self.scale)
    
     def cdf(self, y):
        return (1.0 + m.erf(y / np.sqrt(2.0))) / 2.0
     
     def squared_norm(self):
        return 1/(2**self.scale*np.sqrt(np.pi))
    
    
class LaplaceDistribution(Distribution) :
    
     def __init__(self, loc = 0.0, scale =1.0):
         self.loc = loc 
         self.scale = scale
         
     def name(self):
         return "laplace"
     
     def mean(self):
        return self.loc
    
     def pdf(self, y):
        return  np.exp(-np.abs(y -self. loc)/self.scale)/(2*self.scale)
    
     def cdf(self, y):
         if y < self.loc :
             return 0.5 * np.exp((y -self. loc)/self.scale)
         else:
             return 1 - 0.5 * np.exp(-(y -self. loc)/self.scale)
         
     def squared_norm(self):
        return 1/(4*self.scale)