# LEGACY MODULE - TODO: remove or refactor

from skpro.workflow.base import Model


class ModelManager:
    """Model manager"""

    def __init__(self):
        self.models = {}

    def register(self, model, tuning=None, group=None, distinguish_tuning=True):
        model = Model(model, tuning, group)

        self.models[model.identifier(distinguish_tuning)] = model

    def group(self, name):
        if isinstance(name, (tuple, list)):
            models = []
            for group in name:
                models = models + self.group(group)
            return models

        models = []
        for k, v in self.models.items():
            if v.group == name:
                models.append(v.clone())

        return models

    def all(self):
        return [v.clone() for v in self.models.values()]

    def info(self):
        print("Total number of registered models: %i\n" % len(self.models))

        for k, v in self.models.items():
            print(k, str(v))

    def __iter__(self):
        for k, v in self.models.items():
            yield v.clone()

    def __getitem__(self, item):
        if item not in self.models:
            return False

        return self.models[item].clone()

    def __contains__(self, item):
        return item in self.models
