
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics.scorer import check_scoring

import pandas as pd

class ModelManager:
    
    def __init__(self, dataManager, scorer = None):
        
        self.models = {}
        self.scoreResults = {}
        self.CV = {}

        self.dataManager = dataManager
        self.isFit = False
         
       
    def register(self, name, model, fitting_arguments = {}):
        self.models[name] = model
        
        if(len(fitting_arguments) != 0) :
            self.fitArgs[name] = fitting_arguments
        

    def fit(self):
        
        for key, model in self.models.items():
            model.fit(self.dataManager.X_train, self.dataManager.y_train)
        
        self.isFit = True


    def score(self, scoring = None):

        if self.isFit == False :
            self.fit()

        for key, model in self.models.items():
            scoring = check_scoring(model, scoring)
            train_score = scoring(model, self.dataManager.X_train, self.dataManager.y_train)
            test_score =  scoring(model, self.dataManager.X_test, self.dataManager.y_test)
            self.scoreResults[key] = {'train_score':train_score, 'test_score':test_score}
        
        return pd.DataFrame.from_dict(self.scoreResults).transpose()


    def cross_val_score(self, n_splits=5, n_repeats=1, random_state=2652124, scoring = None):
        
        self.__initCV()
        rkf = RepeatedKFold(n_splits, n_repeats, random_state)
        num_ = 0
        
        for fold, (train_index, test_index) in enumerate(rkf.split(self.dataManager.X)):
            X_train, X_test = self.dataManager.X[train_index], self.dataManager.X[test_index]
            y_train, y_test = self.dataManager.y[train_index], self.dataManager.y[test_index]

            for key, model in self.models.items() :
                scoring = check_scoring(model, scoring)
                model.fit(X_train, y_train)
                score = scoring(model, X_test, y_test)
               
                ret = {'score': score, 
                        'trained_index' : train_index, 
                        'trained_params' : model.get_params()
                       }
 
                n = len(X_test)
                weight = n/(num_ + n)
                self.CV[key][fold] = ret
                self.CV[key]['Total'] += weight*(score - self.CV[key]['Total'])
            
            num_ += n

        ret = pd.DataFrame([(key, self.CV[key]['Total']) for key in self.CV], columns = ['model', 'score'])
        return ret

                
    def __initCV(self):
        self.CV = {}
        
        for key, model in self.models.items() :
            self.CV[key] = {'Total': 0}
    
