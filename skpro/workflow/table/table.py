import abc
import numpy as np
from tabulate import tabulate

from ..base import Model, View, Controller
from ..utils import ItemView


class Modifier(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def modify(self, raw, headers):

        return raw, headers


class IdModifier(Modifier):

    def modify(self, raw, headers):
        for i, row in enumerate(raw):
            raw[i] = [{'data': {'index': i + 1}, 'view': ItemView('index')}] + row

        headers = ['#'] + headers

        return raw, headers


class RankModifier(Modifier):

    def __init__(self, vertical='score', horizontal='score', visible=False):
        self.vertical = self.conf(vertical)
        self.horizontal = self.conf(horizontal)
        self.visible = visible
        self.rendered_ = None

    def conf(self, value):
        # TODO: find better solution
        if value is False:
            return False, False, False
        elif value[0] == '~':
            return value[1:], True, False
        elif value[0] == '*':
            return value[1:], True, True
        else:
            return value, False, False

    def rankdata(self, a):
        """
        https://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        :param a:
        :return:
        """
        def rank_simple(vector):
            return sorted(range(len(vector)), key=vector.__getitem__)

        n = len(a)
        ivec = rank_simple(a)
        svec = [a[rank] for rank in ivec]
        sumranks = 0
        dupcount = 0
        newarray = [0] * n
        for i in range(n):
            sumranks += i
            dupcount += 1
            if i == n - 1 or svec[i] != svec[i + 1]:
                averank = sumranks / float(dupcount) + 1
                for j in range(i - dupcount + 1, i + 1):
                    newarray[ivec[j]] = averank
                sumranks = 0
                dupcount = 0
        return newarray

    def modify(self, raw, headers):

        # Calculate horizontal ranks
        if self.horizontal[0]:
            field = self.horizontal[0]
            for row in raw:
                sortable = []
                map = {}
                for i, cell in enumerate(row):
                    if field in cell['data']:
                        sortable.append(cell['data'][field])
                        map[len(sortable) - 1] = cell['data']

                for k, rank in enumerate(self.rankdata(sortable)):
                    map[k]['hrank'] = rank


        # Calculate vertical ranks
        if self.vertical[0]:
            field = self.vertical[0]
            for col in range(len(raw[0])):
                sortable = []
                map = {}
                for i in range(len(raw)):
                    cell = raw[i][col]
                    if field in cell['data']:
                        sortable.append(cell['data'][field])
                        map[len(sortable) - 1] = cell['data']

                for k, rank in enumerate(self.rankdata(sortable)):
                    map[k]['vrank'] = rank

        # aggregate
        if self.vertical[0] and self.vertical[1]:
            for row in raw:
                avg = []
                for cell in row:
                    # TODO
                    # Using 'vrank' as a key in a lot of plases.
                    # May be better to define it in a variable?
                    if 'vrank' in cell['data']:
                        avg.append(cell['data']['vrank'])

                avg_mean = np.mean(avg) if len(avg) > 0 else 0
                row.append({
                    'data': {'vrank': avg_mean}
                })

        if self.horizontal[0] and self.horizontal[1]:
            avgs = []
            for col in range(len(raw[0])):
                avg = []
                for i in range(len(raw)):
                    cell = raw[i][col]
                    if 'hrank' in cell['data']:
                        avg.append(cell['data']['hrank'])
                avg_mean = np.mean(avg) if len(avg) > 0 else 0
                avgs.append({
                    'data': {'hrank': avg_mean}
                })
            raw.append(avgs)

        return raw, headers


class SortModifier(Modifier):

    def __init__(self, key, reverse=False):
        self.key = key
        self.reverse = reverse

    def modify(self, raw, headers):
        sorted_raw = sorted(raw, key=self.key, reverse=self.reverse)
        return sorted_raw, headers


class Table:

    def __init__(self, tasks=None, modifiers=None):
        if tasks is None:
            tasks = {}
        self.tasks = tasks
        if modifiers is None:
            modifiers = {}
        self.modifiers = modifiers
        self.rendered_ = None

    def add(self, controller, view):
        if not issubclass(controller.__class__, Controller):
            raise Exception('controller has to be subclass instance skpro.workflow.Controller')

        if not issubclass(view.__class__, View):
            raise Exception('view has to be subclass instance of skpro.workflow.View')

        self.tasks.append((controller, view))
        self.rendered_ = None

    def modify(self, modifier):
        self.modifiers.append(modifier)

    def render(self, models, verbose=0, debug=False):
        if len(self.tasks) == 0:
            raise Exception('The table is empty. You have to include tasks using add()')

        # Compose table
        raw_tbl = []
        headers = []
        i = 0
        for model in models:
            if not issubclass(model.__class__, Model):
                raise Exception('models must only contain subclass instances of skpro.workflow.Model')

            if verbose > 1:
                print('Evaluating model %i/%i: %s' % (i + 1, len(models), str(model)))
            row = []
            for controller, view in self.tasks:
                controller_id = controller.identifier()
                if not debug or str(controller) == 'Info':
                    model[controller_id] = controller.run(model)
                else:
                    model[controller_id] = {'debug': 'debug'}
                    view = ItemView('debug')

                if verbose > 2:
                    print('Running ' + str(controller))
                    print(model[controller_id])

                if i == 0:
                    headers.append(str(controller))

                row.append({
                    'data': model[controller_id],
                    'view': view
                })
            i += 1
            raw_tbl.append(row)

        # Apply modifiers
        for modifier in self.modifiers:
            raw_tbl, headers = modifier.modify(raw_tbl, headers)

        # Parse raw table
        tbl = []
        for raw in raw_tbl:
            tbl.append([
                cell['view'].parse(cell['data'])
                for cell in raw if 'view' in cell
            ])

        self.rendered_ = {
            'raw': raw_tbl,
            'parsed': tbl,
            'headers': headers
        }

        return self.rendered_

    def print(self, models, fmt='pipe', with_headers=True, raw=False, return_only=False, verbose=1, debug=False):
        """
        Renders and prints the table
        :param fmt: None, 'pipe', 'latex'
        :param with_headers: Bool
        :param return_only:
        :return:
        """
        self.render(models, verbose=verbose, debug=debug)

        if with_headers:
            headers = self.rendered_['headers']
        else:
            headers = []

        if raw:
            rendered = self.rendered_['raw']
        else:
            rendered = self.rendered_['parsed']

        if isinstance(fmt, str):
            fmt = [fmt]

        tbl = ''
        for f in fmt:
            if f == 'raw':
                f = 'pipe'
                rendered = self.rendered_['raw']

            tbl += '\n' + tabulate(
                rendered,
                headers=headers,
                tablefmt=f,
            ) + '\n'

        if not return_only:
            print(tbl)

        return tbl
