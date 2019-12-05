# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 04:50:16 2019

@author: jesel
"""

import numpy as np

class VectorizationMixin():
    
    
    def _vectorize(self, f, y = None, mode = 'element_wise'):
        
        self._checkMode(mode)
        
        if y is None :
            return self._nullpointWise(f) 
        elif self._isScalar(y):  
            return self._pointWise(f, y) 
        elif mode == 'element_wise' : 
            return self. _elementWise(f, y)
        else: return self. _batchWise(f, y)
        
    

    def _pointWise(self, f, y):
        
        if(self.size_ == 1):
            return f(y, self.get_params(0))
 
        out = []
        for k in range(self.size_):
            params = self.get_params(k)
            out.append(f(y, params))

        return out
    
    
    def _nullpointWise(self, f):
        
        if(self.size_ == 1):
            return f(self.get_params(0))
 
        out = []
        for k in range(self.size_):
            params = self.get_params(k)
            out.append(f(params))
            
        return out

        
    def _elementWise(self, f, y):

        out = []
        for k in range(len(y)):
            index = k%self.size_
            params = self.get_params(index)
            out.append(f(y[k], params))

        return out
    
    
    def _batchWise(self, f, y):

        out = []
        for index in range(self.size_):
            tmp = []
            params = self.get_params(index)
            
            for k in range(len(y)):
                tmp.append(f(y[k], params))

            out.append(tmp)
        
        return out
    
    
    def _checkMode(self, mode):
        allowed_mode = ("element_wise", "batch_wise")
        if mode not in allowed_mode:
            raise ValueError("Unknown mode type: %s, expected one of %s."
                             % (mode, allowed_mode))
            
    def _isScalar(self, y):
        
        if np.isscalar(y) : return True
        elif(len(np.shape(y)) == 1 and self.dimension_ == len(y)) : return True
        
        return False
        
        
            