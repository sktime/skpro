# LEGACY MODULE - TODO: remove or refactor

from .base import Controller, View


class RawView(View):
    """Raw view"""

    def parse(self, data):
        return str(data)


class ItemView(View):
    """Item view"""

    def __init__(self, key):
        self.key = key

    def parse(self, data):
        if self.key in data:
            return str(data[self.key])
        else:
            return self.key


class InfoView(View):
    """InfoView"""

    def __init__(self, key="description", with_group=False):
        self.key = key
        self.with_group = with_group

    def parse(self, data):
        if self.with_group:
            return "%s [%s]" % (data[self.key], data["group"])

        return data[self.key]


class InfoController(Controller):
    """InfoController"""

    def identifier(self):
        return "InfoController()"

    def __repr__(self):
        return "InfoController()"

    def __str__(self):
        return "Info"

    def run(self, model):
        return {
            "description": str(model).replace("\n           ", ""),
            "repr": repr(model),
            "group": getattr(model, "group", "None"),
        }
