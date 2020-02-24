
import pandas as pd


class parametersFrame :
    """ Class parametersFrame : Wrap-up object of a pandas DataFrame object
        It is meant to be used as the main PARAMETER CONTAINER for each distribution class
    
        Parameters :
        ----------
        data: dict
             dictionary of parameters {keys, list of values} 
             
        Note
        ---------
        The dict argument that is converted into dataFrame object
        within the __init__ and kept as member (within self.data_).
        
        self.data_ shapes shape = (n_distribution_samples, m_parameters) 

    """
    
    def __init__(self, data = []):
        self.data_ = data
        
    def setData(self, data):
        self.data_ = pd.DataFrame(data)

    def data(self):
        return self.data_
    
    def __len__(self):
        return self.data_.shape[0]
    
    

    def getSubset(self, index = None):
        """Return a subset of the parameters frame according to the index
           Hence sliced on the horizontal axis (i.e. all parameters per indexed distribution subset)

        Parameters
        ----------
        index : signed int or slice object
        
        Return
        ----------
        dict
            dictionary of parameters {keys, lists of values} 
            whith m_parameters keys mapped to a list of 'index' size per key
             
        """
        
        # parse key
        if isinstance(index, str) or index == None:
            index = slice(None)
        
        elif isinstance(index, int):
            if index > len(self) - 1 :
                raise IndexError('index out of bound')
                
            index = slice(index, index)
            return self.data_.loc[index].to_dict('records')[0]
            
        elif isinstance(index, slice) and index.stop > len(self) - 1 :
            raise IndexError('index call out of bound')


        return self.data_.loc[index].to_dict('list')
        

    def getParameter(self, key, mode = 'list'):
        """Return a SINGLE KEYED parameter for ALL distribution
           Hence slice on the vertically axis (one column only)

        Parameters
        ----------
        key : string
              string tag of the parameter to be returned
              
        mode : string
              specify the formating of the output :
              [list] return the parameters as list (use by default)
              [dic] return the parameters as dictionary
        
        Return
        ----------
        A list or dict (depending on the mode selected)
        """
        
        if mode == 'dic' :
            return {key : self.data_[key].tolist()}
        else :
            return self.data_[key].tolist()
        