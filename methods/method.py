import torch
from util import Statistics
from eval import eval

class Method:
    def __init__(self, model, retrain_model, loaders, args):
        self.statistics = Statistics(model, args)
        self.model = model
        self.retrain_model = retrain_model
        self.loaders = loaders
        self.args = args
        if args.unlearn_aug:
            self.train_forget_set = loaders['train_forget_set']
            self.train_remain_set = loaders['train_remain_set']
            self.train_forget_loader = loaders['train_forget_loader']
            self.train_remain_loader = loaders['train_remain_loader']
        else:
            self.train_forget_set = loaders['train_forget_test_set']
            self.train_remain_set = loaders['train_remain_test_set']
            self.train_forget_loader = loaders['train_forget_test_loader']
            self.train_remain_loader = loaders['train_remain_test_loader']
        
    def run(self):
        self.set_hyperparameters()
        self.statistics.start_record()
        model = self.unlearn(self.model, self.loaders, self.args)
        self.statistics.end_record()
        return model, self.statistics

    # Implement unlearning method here
    def unlearn(self, model, loaders, args):
        pass

    # here to distinguish the hyperparameters
    def set_hyperparameters(self):
        pass
    
    def get_optimizer(self, model):
        if self.args.optimizer == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=.9, weight_decay=5e-4)
        else:
            raise ValueError(f"Invalid optimizer: {self.args.optimizer}")
        
    def intermidiate_test(self, model):
        model.eval()
        remain_train_acc = eval(model, self.loaders['train_remain_test_loader'], self.args)
        forget_train_acc = eval(model, self.loaders['train_forget_test_loader'], self.args)
        remain_test_acc = eval(model, self.loaders['test_remain_loader'], self.args)
        forget_test_acc = eval(model, self.loaders['test_forget_loader'], self.args)
        self.args.logger.info(f'Remain Train Acc: {remain_train_acc:.4f}, Forget Train Acc: {forget_train_acc:.4f}, Remain Test Acc: {remain_test_acc:.4f}, Forget Test Acc: {forget_test_acc:.4f}')
        model.train()