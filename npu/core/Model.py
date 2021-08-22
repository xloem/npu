class Model:

    def __init__(self, name):
        self.name = name
        self.attr = ""

    def __call__(self, pretrained=False, *args, **kwargs):
        return Model(self.name + ("_trained" if pretrained else ""))


class NLPHFModel(Model):

    def __init__(self, name, head):
        Model.__init__(self, name)
        self.attr = "{}_{}".format(head, name)


def nlp_constructor(head, pretrained):
    return NLPHFModel(name=pretrained, head=head)
