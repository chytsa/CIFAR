import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import pylab
import torch.optim as optim     #导入torch.potim模块
""" ============================================================================================ """
# 首先是调用Variable、 torch.nn、torch.nn.functional
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):                           # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):    
        super(Net, self).__init__()             # 第二、三行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
        self.conv1 = nn.Conv2d(3, 6, 5)         # 添加第一个卷积层,调用了nn里面的Conv2d（）
                                                # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
                                                
        self.pool = nn.MaxPool2d(2, 2)          # 最大池化层
                                                # (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv2 = nn.Conv2d(6, 16, 5)        # 同样是卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # 接着三个全连接层
                                                # (in_features, out_features, bias=True)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
 
    def forward(self, x):                       # 这里定义前向传播的方法，为什么没有定义反向传播的方法呢？这其实就涉及到torch.autograd模块了，
                                                # 但说实话这部分网络定义的部分还没有用到autograd的知识，所以后面遇到了再讲
        x = self.pool(F.relu(self.conv1(x)))    # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
                                    #  第一个参数-1是说这个参数由另一个参数确定， 比如矩阵在元素总数一定的情况下，确定列数就能确定行数。
                                    #  那么为什么这里只关心列数不关心行数呢，因为马上就要进入全连接层了，而全连接层说白了就是矩阵乘法，
                                    #  你会发现第一个全连接层的首参数是16*5*5，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
                                    # 更多的Tensor方法参考Tensor: http://pytorch.org/docs/0.3.0/tensors.html
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
""" ============================================================================================ """
def main():
    # 定义是否使用GPU
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #  首先当然肯定要导入torch和torchvision，至于第三个是用于进行数据预处理的模块
    #  **由于torchvision的datasets的输出是[0,1]的PILImage，所以我们先先归一化为[-1,1]的Tensor**
    #  首先定义了一个变换transform，利用的是上面提到的transforms模块中的Compose( )
    #  把多个变换组合在一起，可以看到这里面组合了ToTensor和Normalize这两个变换
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
 
    # 定义了我们的训练集，名字就叫trainset，至于后面这一堆，其实就是一个类：
    # torchvision.datasets.CIFAR10( )也是封装好了的，就在我前面提到的torchvision.datasets
    # 模块中,不必深究，如果想深究就看我这段代码后面贴的图1，其实就是在下载数据
    #（不翻墙可能会慢一点吧）然后进行变换，可以看到transform就是我们上面定义的transform
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    # trainloader其实是一个比较重要的东西，我们后面就是通过trainloader把数据传入网
    # 络，当然这里的trainloader其实是个变量名，可以随便取，重点是他是由后面的
    # torch.utils.data.DataLoader()定义的，这个东西来源于torch.utils.data模块，
    #  网页链接http://pytorch.org/docs/0.3.0/data.html，这个类可见我后面图2
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,
                                          shuffle=True, num_workers=2)
    # 对于测试集的操作和训练集一样，我就不赘述了
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=False, num_workers=2)
    # 类别信息也是需要我们给定的
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
    # 和python中一样，类定义完之后实例化就很简单了，我们这里就实例化了一个net
    net = Net()#.to(device)
    """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
    criterion = nn.CrossEntropyLoss()                               #同样是用到了神经网络工具箱 nn 中的交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9) #optim模块中的SGD梯度优化方式---随机梯度下降
    
    epoch = 20
    train(trainloader, testloader, net, criterion, optimizer, epoch)
    print('Validation:')
    validate(testloader, net, criterion)
    
""" ============================================================================================ """
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def train(trainloader, testloader, net, criterion, optimizer, epoch):
    print('Train #epoch: ',epoch,' Start')
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.train()

    for ep in range(epoch):                         # 指定训练一共要循环几个epoch
        
        running_loss = 0.0                          #定义一个变量方便我们对loss进行输出
        for i, data in enumerate(trainloader, 0):   # 这里我们遇到了第一步中出现的trailoader，代码传入数据
                                                    # enumerate是python的内置函数，既获得索引也获得数据，详见下文
            # get the inputs
            inputs, labels = data                   # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels

            #inputs, labels = inputs.to(device), labels.to(device)
                                                                # 所以这段程序里面就直接使用了，下文会分析
            # zero the parameter gradients
            optimizer.zero_grad()                   # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度
     
            # forward + backward + optimize      
            outputs = net(inputs)                   # 把数据输进网络net，这个net()在第二步的代码最后一行我们已经定义了
            loss = criterion(outputs, labels)       # 计算损失值,criterion我们在第三步里面定义了
            
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
            loss.backward()                         # loss进行反向传播，下文详解
            optimizer.step()                        # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
     
            # print statistics                      # 这几行代码不是必须的，为了打印出loss方便我们看而已，不影响训练过程
            running_loss += loss.item()             # 从下面一行代码可以看出它是每循环0-1999共两千次才打印一次
            if (i + 1) % 250 == 0:                    # print every 2000 mini-batches   所以每个2000次之类先用running_loss进行累加
                """print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                                                    # 然后再除以2000，就得到这两千次的平均损失值
                running_loss = 0.0                  # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用"""
                print('Epoch: {0} [{1}/{2}]\n'
                  'min-batch avg loss:{loss.avg:.4f}\t'
                  'top-1 accuracy:{top1.avg:.3f}\t'
                  'top-5 accuracy:{top5.avg:.3f}'.format(
                   (ep + 1), (i + 1), len(trainloader), loss=losses, top1=top1, top5=top5))
        print('')
        if ep == 7:
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        elif ep == 12:
            optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
        validate(testloader, net, criterion)
    print('Finished Training')
        
def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            #input = input.cuda(args.gpu, non_blocking=True)
            #target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            if (i + 1) % 50 == 0:
                print('Test: [{0}/{1}]\n'
                      'min-batch avg loss:{loss.avg:.4f}\t'
                      'top-1 accuracy:{top1.avg:.3f}\t'
                      'top-5 accuracy:{top5.avg:.3f}'.format(
                       (i + 1), len(val_loader), loss=losses,
                       top1=top1, top5=top5))


        print('\ntop-1:{top1.avg:.3f}\ttop-5:{top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
""" ============================================================================================ """
if __name__ == '__main__':
    main()
