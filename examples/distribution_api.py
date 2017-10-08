import numpy as np
from matplotlib import pyplot




class Dist:

    def __init__(self):
        self.a = np.arange(10)
        self.value = None

    @classmethod
    def clone(cls):
        return cls()

    def __getitem__(self, item):
        print(item)
        #return T(self.a[item])

    def __len__(self):
        return len(self.a)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

y_pred = Dist()

y_pred[:, 'hello']
exit()

y = np.arange(10)+1

pyplot.scatter(y, y_pred)
pyplot.show()