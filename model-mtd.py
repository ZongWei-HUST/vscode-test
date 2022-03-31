from timm import create_model
from torch import device
import torch
from timm.data.transforms_factory import create_transform
from timm.data import ImageDataset
from torch.utils.data import DataLoader
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler, StepLRScheduler
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from timm.loss import AsymmetricLossMultiLabel, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from custom_utils.focalloss import FocalLoss
from custom_utils.seesawloss import DistibutionAgnosticSeesawLossWithLogits
from timm.data.mixup import Mixup


device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model_name = "vit_base_patch16_224"
model = create_model(model_name, pretrained=True, num_classes=6)
net = model.to(device)


def create_dataloader_iterator():
    trans = create_transform(224, interpolation="bicubic")
    # trans = create_transform(224, interpolation="bicubic", auto_augment="rand-m9-mstd0.5", is_training=True)
    
    train_dataset = ImageDataset("Projects/MTDClassification/dataset/Magnetic_Tile_Defect/train", transform=trans)
    test_dataset = ImageDataset("Projects/MTDClassification/dataset/Magnetic_Tile_Defect/val", transform=trans)
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_dl  = DataLoader(test_dataset, batch_size=32)
    
    return train_dl, test_dl, trans

def create_optimizer(t_init=30):
    optimizer = create_optimizer_v2(net.parameters(), opt="sgd", lr=0.001)
    # scheduler = CosineLRScheduler(optimizer, t_initial=t_init, warmup_lr_init=0.0001, warmup_t=3)
    scheduler = StepLRScheduler(optimizer, warmup_lr_init=0.0001, warmup_t=3, decay_t=10)
    return optimizer, scheduler

def creat_loss():
    # loss = CrossEntropyLoss()
    loss = FocalLoss(alpha=0.3)
    # loss = AsymmetricLossMultiLabel() # need target -> (batch, num_classes), one-hot
    # loss = LabelSmoothingCrossEntropy()
    # loss = DistibutionAgnosticSeesawLossWithLogits(num_labels=6)
    # loss = SoftTargetCrossEntropy() # be used with mixup 
    # loss = JsdCrossEntropy(num_splits=2,smoothing=0,alpha=0) # be used with autoaug, num_splits=变化增强混合数
    return loss

if __name__ == '__main__':
    # load dataset
    train_dl, test_dl, trans = create_dataloader_iterator()
    optimizer, scheduler = create_optimizer()
    loss_fn = creat_loss()
    print(trans)

    # epochs
    train_epochs = 25
    cooldowm_epochs = 10
    num_epochs = train_epochs + cooldowm_epochs
    
    # if mixup
    mixup = False
    mixup_args = {
        'mixup_alpha': 0.3,
        'cutmix_alpha': 0.3,
        'cutmix_minmax': None,
        'prob': 1.0,
        'switch_prob': 0.,
        'mode': 'batch',
        'label_smoothing': 0,
        'num_classes': 6
    }
    mixup_fn = Mixup(**mixup_args)
    
    # train
    net.train()
    for epoch in range(num_epochs):
        print("Epoch {}\n-------------------------------".format(epoch + 1))
        train_step = 0
        num_steps_per_epoch = len(train_dl)
        num_updates = epoch * num_steps_per_epoch
        
        for batch in train_dl:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            if mixup:
                inputs, targets = mixup_fn(inputs, targets) # targets will be [batch_size, num_classes]

            outputs = net(inputs)
            # targets = F.one_hot(targets, num_classes=6) # asy loss
            # print(outputs.dtype, targets.dtype)
            loss = loss_fn(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            scheduler.step_update(num_updates=num_updates)

            train_step += 1
            if train_step % 5 == 0:
                print("step: {} loss: {}".format(train_step, loss.item()))
        
        scheduler.step(epoch + 1)
        
        # evaluate
        net.eval()
        num_batchs = len(test_dl)
        size = len(test_dl.dataset)
        # print(num_batchs, size)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for inputs_test, targets_test in test_dl:
                inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
                pred = net(inputs_test)

                # print(pred, targets_test)
                # targets_test = F.one_hot(targets_test, num_classes=6) # asy loss
                test_loss += loss_fn(pred, targets_test).item()
                correct   += (pred.argmax(1) == targets_test).type(torch.float).sum().item()
                # correct   += (pred.argmax(1) == targets_test.argmax(1)).type(torch.float).sum().item()
        test_loss /= num_batchs
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
save_path = "Projects/MTDClassification/work_dir/" + model_name + ".pth"
torch.save(net, save_path)
