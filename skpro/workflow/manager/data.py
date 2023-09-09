# LEGACY MODULE - TODO: remove or refactor

import hashlib
import os
import urllib.request

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def retrieve_data(url):
    file_name = "/tmp/%s.dat" % hashlib.md5(str(url).encode("utf-8")).hexdigest()
    if not os.path.isfile(file_name):
        with urllib.request.urlopen(url) as response, open(file_name, "wb") as out_file:
            data = response.read()
            out_file.write(data)

    return file_name


def load_file(file, return_X_y=False):
    data = np.loadtxt(file)

    if return_X_y:
        return data[:, :-1], data[:, -1:].ravel()
    else:
        return data


def load_url(url, return_X_y=False):
    file = retrieve_data(url)
    return load_file(file, return_X_y)


class DataManager:
    """
    A helper to manage datasets more easily. Test/training split
    is carried out behind the scenes whenever new data is being
    assigned

    Parameters
    ----------
    X : np.array | string
        Features or 'boston', 'diabetes' to load sklearn datasets, url of file
    y : np.array
        Labels
    split: float, default=0.2
        Train/test split
    name: string, default=None
        Optional name to be used in the object representation
    random_state: int, default=None
        Optional random state to be used during split

    Attributes
    ----------
    X_train : np.array
        Training features
    X_test : np.array
        Training labels
    y_train : np.array
        Test features
    y_test : np.array
        Test labels
    """

    def __init__(self, X=None, y=None, split=0.2, name=None, random_state=None):
        if isinstance(X, str):
            # autoload sklearn datasets, urls and files
            name = X
            if name.lower() == "boston":
                X, y = load_diabetes(return_X_y=True)
            elif name.lower() == "diabetes":
                X, y = load_diabetes(return_X_y=True)
            elif name.startswith("http"):
                X, y = load_url(name, return_X_y=True)
            elif os.path.isfile(name):
                X, y = load_file(name, return_X_y=True)
            else:
                raise ValueError("'%s' is not a valid dataset" % X)

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
        self.name = name if isinstance(name, str) else "Unnamed"

    def data(self, copy=True):
        """Returns the data

        Parameters
        ----------
        copy: boolean, default=True
            If false, reference copy will be used

        Returns
        -------
        X, y
        """
        if not copy:
            return self.X, self.y

        try:
            X_ = np.copy(self._X)
            y_ = np.copy(self._y)
            return X_, y_
        except:
            return False

    def clone(self):
        """Clones the data manager

        Returns
        -------
        A copy of the data manager itself
        """
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
            self._X, self._y, test_size=self.split, random_state=self.random_state
        )

    def shuffle(self, random_state=None):
        """Shuffles the data

        Parameters
        ----------
        random_state: int, default=None
            Optional random state

        Returns
        -------
        None
        """
        if random_state is not None:
            self.random_state = random_state

        data = list(zip(np.copy(self.X), np.copy(self.y)))
        np.random.shuffle(data)
        X, y = zip(*data)

        self.X, self.y = np.array(X), np.array(y)

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
