import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=1, gamma=1.5, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
#            print("self.alpha:", self.alpha)
        else:
            if isinstance(alpha, Variable):
#                print("1111111111111111111111")
                self.alpha = alpha
            else:
#                print("2222222222222222222222")
                self.alpha = Variable(torch.full((class_num, 1), alpha, dtype= torch.float))
                # self.alpha = Variable(alpha)
#                print("self.alpha:", self.alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print("N:", N)
        C = inputs.size(1)
        # print("C:", C)
        # P = F.softmax(inputs, dim=1)
        P = torch.sigmoid_(inputs)
        # print("P:",P)
        class_mask = inputs.data.new(N, C).fill_(0)
        # print("class_mask1:", class_mask)
        class_mask = Variable(class_mask)
        # print("class_mask2:", class_mask)
        ids = targets.view(-1, 1)
        # print("ids:",ids)
        class_mask.scatter_(1, ids.data, 1.)
        # print("class_mask.scatter_:", class_mask.scatter_(1, ids.data, 1.))
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        # print("alpha:", alpha)
        probs = (P*class_mask).sum(1).view(-1,1)
        # print("(P*class_mask).sum(1):",(P*class_mask).sum(1).view(-1,1))
#        print("probs:", probs)
        log_p = probs.log()
        # print("log_p:", log_p)
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
#        print("1-probs:",1-probs)
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
#        print("batch_loss:", batch_loss)
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    
if __name__ == '__main__':
    loss = FocalLoss(class_num = 2)
    predict = torch.tensor([[-0.0665,  1.2863],
        [-1.7105,  0.6629],
        [-0.4579,  0.2342],
        [-0.6699,  1.1567],
        [-0.3429,  0.1121],
        [-0.5634,  0.4262],
        [-0.1727,  1.0969],
        [ 0.3436, -0.4539]])
    print(predict)
    label = torch.tensor([[1, 1, 0, 1, 0, 1, 1, 1]])
    print(label)
    result = loss(predict , label)
    print("result:", result)