# LEGACY MODULE - TODO: remove or refactor

if False:
    import numpy as np
    from uncertainties import ufloat
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.model_selection import KFold

    from ..model_selection import cross_val_score
    from ..metrics import make_scorer

from .base import Controller, View


def grid_optimizer(verbose=0, n_jobs=1):
    def wrapper(model, search_space, scoring, cv):
        return GridSearchCV(
            estimator=model,
            param_grid=search_space,
            scoring=scoring,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    return wrapper


def random_optimizer(n_iter=10, verbose=0, n_jobs=1):
    def wrapper(model, search_space, scoring, cv):
        return RandomizedSearchCV(
            estimator=model,
            param_distributions=search_space,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    return wrapper


class CrossValidationController(Controller):
    """CrossValidation controller

    Parameters
    ----------
    data
    loss_func
    cv
    tune
    optimizer: optional callable(model, search_space, scoring, cv), default=None
        Callable that returns an optimizer such as RandomizedSearchCV etc. that
        will be used for the hyperparameter search.
    """

    def __init__(self, data, loss_func, cv=None, tune=False, optimizer=None):
        self.data = data
        self.loss_func = loss_func
        if cv is None:
            cv = KFold()
        self.cv = cv
        self.tune = tune
        if optimizer is None:
            optimizer = grid_optimizer()
        self.optimizer = optimizer

    def identifier(self):
        return "CrossValidation(data=%s, loss_func=%s)" % (
            self.data.name,
            self.loss_func.__name__,
        )

    def __repr__(self):
        return "CrossValidation(data=%s, loss_func=%s, cv=%s, tune=%s)" % (
            self.data.name,
            self.loss_func.__name__,
            repr(self.cv),
            str(self.tune),
        )

    def description(self):
        return "CV(%s, %s)" % (self.data.name, self.loss_func.__name__)

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
            clf = self.optimizer(
                model=model.instance,
                search_space=tuning,
                scoring=make_scorer(self.loss_func, greater_is_better=False),
                cv=self.cv,
            )

            best_params = []
            loss_function = self.loss_func

            def scorer(estimator, X_test, y_test, **kwargs):
                y_pred = estimator.predict(X_test)
                best_params.append(estimator.best_params_)
                return loss_function(y_test, y_pred, **kwargs)

        scores = cross_val_score(
            clf, self.data.X, self.data.y, scoring=scorer, cv=self.cv
        )

        score = ufloat(np.mean(scores[:, 0]), np.mean(scores[:, 1]))

        return {
            "score": score,
            "scores": scores,
            "tuning": tuning,
            "best_params": best_params,
        }


class CrossValidationView(View):
    """Cross validation view

    Parameters
    ----------
    with_tuning
    with_ranks
    """

    def __init__(self, with_tuning=False, with_ranks=True):
        self.with_tuning = with_tuning
        self.with_ranks = with_ranks

    def parse(self, data):
        result = str(data["score"])

        if data["tuning"] is not None:
            if self.with_tuning:
                best = data["best_params"][0]
                tuning = ""
                comma = ""
                for k, v in data["tuning"].items():
                    tuning += comma + k + ": "
                    comma = "; "

                    sep = ""
                    for e in v:
                        t = "%s"
                        if e == best[k]:
                            t = "**%s**"

                        tuning += sep + (t % e)
                        sep = ", "

                result += "* (%s)" % tuning
            else:
                result += "*"

        if self.with_ranks and "vrank" in data:
            result = "(%i) " % data["vrank"] + result

        return result
