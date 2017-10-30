import numpy as np
from uncertainties import ufloat
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from ..model_selection import cross_val_score
from ..metrics import make_scorer
from .base import Controller, View


class CrossValidationController(Controller):
    """ CrossValidation controller

    Parameters
    ----------
    data
    loss_func
    cv
    tune
    """

    def __init__(self, data, loss_func, cv=KFold(), tune=False):
        self.data = data
        self.loss_func = loss_func
        self.tune = tune
        self.cv = cv

    def identifier(self):
        return 'CrossValidation(data=%s, loss_func=%s)' % (self.data.name, self.loss_func.__name__)

    def __repr__(self):
        return 'CrossValidation(data=%s, loss_func=%s, cv=%s, tune=%s)' % (self.data.name, self.loss_func.__name__, repr(self.cv), str(self.tune))

    def description(self):
        return 'CV(%s, %s)' % (self.data.name, self.loss_func.__name__)

    def run(self, model):

        if self.tune:
            tuning = model.tuning
        else:
            tuning = None

        if tuning is None:
            clf = model.instance
            best_params = None
            scorer = make_scorer(self.loss_func)
        else:
            clf = GridSearchCV(model.instance, param_grid=tuning, scoring=make_scorer(self.loss_func, greater_is_better=False), verbose=0, cv=self.cv)

            best_params = []
            loss_function = self.loss_func

            def scorer(estimator, X_test, y_test, **kwargs):
                y_pred = estimator.predict(X_test)
                best_params.append(estimator.best_params_)
                return loss_function(y_test, y_pred, **kwargs)

        scores = cross_val_score(
            clf,
            self.data.X,
            self.data.y,
            scoring=scorer,
            cv=self.cv
        )

        score = ufloat(np.mean(scores[:, 0]), np.mean(scores[:, 1]))

        return {
            'score': score,
            'scores': scores,
            'tuning': tuning,
            'best_params': best_params
        }


class CrossValidationView(View):
    """ Cross validation view

    Parameters
    ----------
    with_tuning
    with_ranks
    """
    def __init__(self, with_tuning=False, with_ranks=True):
        self.with_tuning = with_tuning
        self.with_ranks = with_ranks

    def parse(self, data):
        result = str(data['score'])

        if data['tuning'] is not None:
            if self.with_tuning:
                best = data['best_params'][0]
                tuning = ''
                comma = ''
                for k, v in data['tuning'].items():
                    tuning += comma + k + ': '
                    comma = '; '

                    sep = ''
                    for e in v:
                        t = '%s'
                        if e == best[k]:
                            t = '**%s**'

                        tuning += sep + (t % e)
                        sep = ', '

                result += '* (%s)' % tuning
            else:
                result += '*'

        if self.with_ranks and 'vrank' in data:
            result = '(%i) ' % data['vrank'] + result

        return result
