import abc
import numpy as np
from tabulate import tabulate

from ..base import Model, View, Controller
from ..utils import ItemView, InfoController, InfoView
from ..cross_validation import CrossValidationView, CrossValidationController


class Modifier(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def modify(self, raw, headers):

        return raw, headers


class IdModifier(Modifier):

    def __init__(self, start_with=1):
        self.start_with = start_with

    def modify(self, raw, headers):
        for i, row in enumerate(raw):
            raw[i] = [{'data': {'index': self.start_with + i}, 'view': ItemView('index')}] + row

        headers = ['#'] + headers

        return raw, headers


class RankModifier(Modifier):

    def __init__(self, vertical='score', horizontal='score', aggregate=False, visible=False):
        self.vertical = vertical
        self.horizontal = horizontal
        self.aggregate = aggregate
        self.visible = visible
        self.rendered_ = None

    def _rank(self, a):
        """
        Calculates a rank vector from a list

        Parameters
        ----------
        a : array-like
            The list

        Returns
        -------
        Rank vector

        Additional Information
        ----------------------
        See https://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        """
        def rank_simple(vector):
            return sorted(range(len(vector)), key=vector.__getitem__)

        n = len(a)
        ivec = rank_simple(a)
        svec = [a[rank] for rank in ivec]
        sumranks = 0
        dupcount = 0
        result = [0] * n
        for i in range(n):
            sumranks += i
            dupcount += 1
            if i == n - 1 or svec[i] != svec[i + 1]:
                averank = sumranks / float(dupcount) + 1
                for j in range(i - dupcount + 1, i + 1):
                    result[ivec[j]] = averank
                sumranks = 0
                dupcount = 0

        return result

    def modify(self, raw, headers):
        """

        Parameters
        ----------
        raw
        headers

        Returns
        -------

        """

        # Calculate horizontal ranks
        if self.horizontal:
            field = self.horizontal
            for row in raw:
                sortable = []
                map = {}
                for i, cell in enumerate(row):
                    if field in cell['data']:
                        sortable.append(cell['data'][field])
                        map[len(sortable) - 1] = cell['data']

                for k, rank in enumerate(self._rank(sortable)):
                    map[k]['hrank'] = rank

        # Calculate vertical ranks
        if self.vertical:
            field = self.vertical
            for col in range(len(raw[0])):
                sortable = []
                map = {}
                for i in range(len(raw)):
                    cell = raw[i][col]
                    if field in cell['data']:
                        sortable.append(cell['data'][field])
                        map[len(sortable) - 1] = cell['data']

                for k, rank in enumerate(self._rank(sortable)):
                    map[k]['vrank'] = rank

        # Aggregate
        if self.aggregate:
            # vertical
            for row in raw:
                avg = []
                for cell in row:
                    if 'vrank' in cell['data']:
                        avg.append(cell['data']['vrank'])

                avg_mean = np.mean(avg) if len(avg) > 0 else 0
                row.append({
                    'data': {'vrank': avg_mean}
                })

            # horizontal
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

    def __init__(self, key=None, reverse=False):
        if key is None:
            key = lambda x: x[-1]['data']['score']
        self.key = key
        self.reverse = reverse

    def modify(self, raw, headers):
        sorted_raw = sorted(raw, key=self.key, reverse=self.reverse)
        return sorted_raw, headers


def filter_modifier(modifier):
    if modifier == 'rank' or modifier == 'ranks':
        return RankModifier()
    elif modifier == 'id' or modifier == 'ids':
        return IdModifier()
    elif modifier == 'sort':
        return SortModifier()
    else:
        return modifier


class Table:

    def __init__(self, tasks=None, modifiers=None):
        if tasks is None:
            tasks = []
        self.tasks = tasks
        if modifiers is None:
            modifiers = ['id', 'rank', 'sort']
        # resolve shorthands
        modifiers = [filter_modifier(modifier) for modifier in modifiers]

        self.modifiers = modifiers
        self.rendered_ = None

    def add(self, controller, view):
        if not issubclass(controller.__class__, Controller):
            raise Exception('controller has to be subclass instance skpro.workflow.Controller')

        if not issubclass(view.__class__, View):
            raise Exception('view has to be subclass instance of skpro.workflow.View')

        self.tasks.append((controller, view))
        self.rendered_ = None

        return self

    def cv(self, data, loss_func, tune=False):
        return self.add(CrossValidationController(data, loss_func, tune=tune),
                 CrossValidationView())

    def info(self, with_group=False):
        return self.add(InfoController(), InfoView(with_group=with_group))

    def modify(self, modifier):
        self.modifiers.append(filter_modifier(modifier))

    def render(self, models, verbose=0, debug=False):
        if len(self.tasks) == 0:
            raise Exception('The table is empty. You have to include tasks using add()')

        # Compose table
        raw_tbl = []
        headers = []
        i = 0
        for model in models:
            if not issubclass(model.__class__, Model):
                model = Model(model)

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

        Parameters
        ----------
        models
        fmt: string or list
            None, 'pipe', 'latex'
        with_headers
        raw
        return_only
        verbose
        debug

        Returns
        -------

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
