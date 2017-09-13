from sklearn.model_selection import train_test_split, KFold
import numpy as np


class DataManager:
    """
    A helper to manage datasets more easily. Test/training split
    is carried out behind the scenes whenever new data is being
    assigned
    """

    def __init__(self, X=None, y=None, split=0.2, random_state=None, name='Unnamed'):
        self.random_state = random_state
        self.split = split
        self._X = None
        self._y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = X
        self.y = y
        self.name = name

    def data(self):
        try:
            X_ = np.copy(self._X)
            y_ = np.copy(self._y)
            return X_, y_
        except:
            return False

    def clone(self):
        X, y = self.data()
        return DataManager(X, y, self.split, name=self.name)

    def _split(self):
        if not isinstance(self.split, float):
            return

        if self._X is None or self._y is None:
            return

        if len(self._X) != len(self._y):
            return

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self._X,
            self._y,
            test_size=self.split,
            random_state=self.random_state
        )

    def shuffle(self, random_state=None):
        if random_state is not None:
            self.random_state = random_state

        data = list(zip(np.copy(self.X), np.copy(self.y)))
        np.random.shuffle(data)
        X, y = zip(*data)

        self.X, self.y = np.array(X), np.array(y)

    def kfold(self, splits=3):
        split = self.split
        self.split = False
        kf = KFold(n_splits=splits, shuffle=True, random_state=self.random_state)
        for train, test in kf.split(self.X, self.y):
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.X[train], self.X[test], self.y[train], self.y[test]

            yield True

        self.split = split

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value
        self._split()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self._split()