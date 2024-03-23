from methods.method import Method

class Pretrain(Method):
    def unlearn(self, ori_model, loaders, args):
        return ori_model