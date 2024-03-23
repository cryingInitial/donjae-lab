import torch
import torch.optim as optim
from tqdm import tqdm
from adv_generator import inf_generator
from methods.method import Method

class NegGrad(Method):

    def unlearn(self, model, loaders, args):

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.get_optimizer(model)

        remain_data_gen = inf_generator(self.train_remain_loader)
        forget_data_gen = inf_generator(self.train_forget_loader)

        for itr in tqdm(range(1, args.iter_num+1)):

            x_remain, y_remain = remain_data_gen.__next__()
            x_forget, y_forget = forget_data_gen.__next__()

            x_remain, y_remain = x_remain.to(args.device), y_remain.to(args.device)
            x_forget, y_forget = x_forget.to(args.device), y_forget.to(args.device)

            outputs_forget = model(x_forget.cuda())
            self.statistics.add_forward_flops(x_forget.size(0))

            loss_ascent_forget = -criterion(outputs_forget, y_forget.cuda())

            # Overall loss
            joint_loss = loss_ascent_forget + self.recover(model, x_remain, y_remain, criterion)
            print(joint_loss)
            optimizer.zero_grad()
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            if isinstance(self, NegGrad): 
                self.statistics.add_backward_flops(x_forget.size(0))
            elif isinstance(self, NegGradP):
                self.statistics.add_backward_flops(x_remain.size(0) + x_forget.size(0))

            optimizer.step()

            if itr % args.test_interval == 0: self.intermidiate_test(model)

        return model

    def recover(self, model, x_remain, y_remain, criterion): return 0


class NegGradP(NegGrad):
    def recover(self, model, x_remain, y_remain, criterion):
        outputs_remain = model(x_remain.cuda())
        self.statistics.add_forward_flops(x_remain.size(0))
        loss_descent_remain = criterion(outputs_remain, y_remain.cuda())

        return loss_descent_remain