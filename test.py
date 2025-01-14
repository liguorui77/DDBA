import torch
import torch.nn as nn
import config
from torchvision import transforms
import numpy as np
from PIL import Image
import cupy as cp
import torch.nn as nn
import torch.nn.functional as F
import main
from torch.utils.data import Subset
from torch.utils.data import DataLoader

def Mytest(helper, epoch,
           model, is_poison=False, visualize=False, agent_name_key=""):
    model.eval()
    model = model.cuda()
    total_loss = 0
    correct = 0
    dataset_size = 0
    if helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type']==config.TYPE_SUBIMAGENET \
            or helper.params['type']==config.TYPE_WEBFACE \
            or helper.params['type'] == config.TYPE_TINYIMAGENET:
        data_iterator = helper.test_data
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    total_l = total_loss / dataset_size if dataset_size!=0 else 0

    main.logger.info(' poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format( is_poison, epoch, # model.name,
                                                        total_l, correct, dataset_size,
                                                        acc))
    if visualize: # loss =total_l
        model.test_vis(vis=main.vis, epoch=epoch, acc=acc, loss=None,
                       eid=helper.params['environment_name'],
                       agent_name_key=str(agent_name_key))
    model.train()
    return (total_l, acc, correct, dataset_size)


def Mytest_poison(helper, epoch,
                  model, is_poison=False, visualize=False, agent_name_key=""):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_cpu = torch.device("cpu")
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    # device = "cuda:0"
    # model.to(device)
    model =model.cuda()
    if helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type']==config.TYPE_SUBIMAGENET \
            or helper.params['type']==config.TYPE_WEBFACE \
            or helper.params['type'] == config.TYPE_TINYIMAGENET:
        data_iterator = helper.test_data_poison
        # test_dataset
        class_source_indices = [i for i, (_, label) in enumerate(helper.test_dataset) if label != helper.params['poison_label_swap']] 
        class_source_dataset = Subset(helper.test_dataset, class_source_indices)
        data_iterator = DataLoader(class_source_dataset, batch_size=helper.params['test_batch_size'], shuffle=False, pin_memory = False)
        for batch_id, batch in enumerate(data_iterator):
            # data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=True)
            # data, targets = batch
            data, targets, poison_num = helper.maketrigger_iamge(batch, evaluation=True)
            # data, targets = batch
            # data, targets = data.to(device), targets.to(device)
            
            targets = torch.ones_like(targets) * helper.params['poison_label_swap']
            
            poison_num = len(data)
            data.requires_grad_(False)
            targets.requires_grad_(False)
            poison_data_count += poison_num
            data = data.cuda()
            targets = targets.cuda()
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            

    acc = 100.0 * (float(correct) / float(poison_data_count))  if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0
    main.logger.info(' poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format( is_poison, epoch, # model.name,
                                                        total_l, correct, poison_data_count,
                                                        acc))
    if visualize: #loss = total_l
        model.poison_test_vis(vis=main.vis, epoch=epoch, acc=acc, loss=None, eid=helper.params['environment_name'],agent_name_key=str(agent_name_key))

    model.train()
    return total_l, acc, correct, poison_data_count

