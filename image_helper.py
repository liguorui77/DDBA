from collections import defaultdict
import math
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import torch.utils.data
import csv
from helper import Helper
import main
import random
import logging
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
from models.resnet_cifar import ResNet18
from models.MnistNet import MnistNet
from models.resnet_tinyimagenet import resnet18
from models.resnet_subimagenet import resnet18_sub
from fedlab.utils.dataset.partition import CIFAR10Partitioner
logger = logging.getLogger("logger")
import config
from config import device
import copy
import cv2
import torch.nn.functional as F
import yaml
from PIL import Image
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import datetime
import json
import cupy as cp
from torchvision import datasets, transforms, models
from torch import optim, nn
from models.preact_resnet import PreActResNet18
class ImageHelper(Helper):

    def create_model(self):
        local_model=None
        target_model=None
        if self.params['type']==config.TYPE_CIFAR:
            local_model = ResNet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = ResNet18(name='Target',
                                   created_time=self.params['current_time'])
            
            out_features=10

            local_model = ResNet18(name='Local',
                                   created_time=self.params['current_time'], num_classes=43)
            target_model = ResNet18(name='Target',
                                   created_time=self.params['current_time'], num_classes=43)
           

        elif self.params['type']==config.TYPE_MNIST:
            local_model = MnistNet(name='Local',
                                   created_time=self.params['current_time'])
            target_model = MnistNet(name='Target',
                                    created_time=self.params['current_time'])

        elif self.params['type']==config.TYPE_TINYIMAGENET:

            local_model= resnet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = resnet18(name='Target',
                                    created_time=self.params['current_time'])
            local_model.fc = nn.Linear(local_model.fc.in_features, 200)
            target_model.fc = nn.Linear(local_model.fc.in_features, 200)





            
        elif self.params['type']==config.TYPE_SUBIMAGENET:

            local_model= resnet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = resnet18(name='Target',
                                    created_time=self.params['current_time'])
            local_model.fc = nn.Linear(local_model.fc.in_features, 200)
            target_model.fc = nn.Linear(local_model.fc.in_features, 200)
        elif self.params['type']==config.TYPE_WEBFACE:

            local_model= resnet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = resnet18(name='Target',
                                    created_time=self.params['current_time'])
            local_model.fc = nn.Linear(local_model.fc.in_features, 200)
            target_model.fc = nn.Linear(local_model.fc.in_features, 200)

        local_model=local_model.to(device)
        target_model=target_model.to(device)
        if self.params['resumed_model']:
            if torch.cuda.is_available() :
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}",map_location='cpu')
                  
            #target_model.load_state_dict(loaded_params['state_dict'])
            target_model.load_state_dict(torch.load('./clean_models/model')) # cifar
            # self.start_epoch = loaded_params['epoch']+1
            self.start_epoch = 0
            # self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            self.params['lr'] = 0.001
            # self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model

    def build_classes_dict(self):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        return cifar_classes

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = self.classes_dict
        class_size = len(cifar_classes[0]) #for cifar: 5000
        # for i in range(43):   
        #     print(len(cifar_classes[i]))

        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())  # for cifar: 10
        # print(no_classes)
        image_nums = []
        for n in range(no_classes):
            image_num = []
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
            image_nums.append(image_num)
        # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
        return per_participant_list

    def draw_dirichlet_plot(self,no_classes,no_participants,image_nums,alpha):
        fig= plt.figure(figsize=(10, 5))
        s = np.empty([no_classes, no_participants])
        for i in range(0, len(image_nums)):
            for j in range(0, len(image_nums[0])):
                s[i][j] = image_nums[i][j]
        s = s.transpose()
        left = 0
        y_labels = []
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, no_participants))
        for k in range(no_classes):
            y_labels.append('Label ' + str(k))
        vis_par=[0,10,20,30]
        for k in range(no_participants):
        # for k in vis_par:
            color = category_colors[k]
            plt.barh(y_labels, s[k], left=left, label=str(k), color=color)
            widths = s[k]
            xcenters = left + widths / 2
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            # for y, (x, c) in enumerate(zip(xcenters, widths)):
            #     plt.text(x, y, str(int(c)), ha='center', va='center',
            #              color=text_color,fontsize='small')
            left += s[k]
        plt.legend(ncol=20,loc='lower left',  bbox_to_anchor=(0, 1),fontsize=4) #
        # plt.legend(ncol=len(vis_par), bbox_to_anchor=(0, 1),
        #            loc='lower left', fontsize='small')
        plt.xlabel("Number of Images", fontsize=16)
        # plt.ylabel("Label 0 ~ 199", fontsize=16)
        # plt.yticks([])
        fig.tight_layout(pad=0.1)
        # plt.ylabel("Label",fontsize='small')
        fig.savefig(self.folder_path+'/Num_Img_Dirichlet_Alpha{}.pdf'.format(alpha))

    def poison_test_dataset(self):
        logger.info('get poison test loader')
        # delete the test data with target label
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)
        poison_label_inds = test_classes[self.params['poison_label_swap']]

        return torch.utils.data.DataLoader(self.test_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                               range_no_id)), \
               torch.utils.data.DataLoader(self.test_dataset,
                                            batch_size=self.params['batch_size'],
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                poison_label_inds))
    def load_data(self):
        logger.info('Loading data')
        dataPath = './data'
        if self.params['type'] == config.TYPE_CIFAR:
            ### data load
            transform_train = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

            transform_test = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True,
                                             transform=transform_train)

            self.test_dataset = datasets.CIFAR10(dataPath, train=False, transform=transform_test)
            #debug
            transform_GTSRB = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.RandomHorizontalFlip(),
             transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
             ])
            transform_GTSRB_test = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
             ])
            self.train_dataset = GTSRB(True, transform_GTSRB)
            print(len(self.train_dataset))
            self.test_dataset = GTSRB(False, transform_GTSRB_test)

        elif self.params['type'] == config.TYPE_MNIST:

            self.train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Grayscale(num_output_channels=3),  
                                   transforms.Resize((28, 28)),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
            self.test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Resize((32, 32)),
                    transforms.Grayscale(num_output_channels=3),  
                    transforms.Resize((28, 28)),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
        elif self.params['type'] == config.TYPE_TINYIMAGENET:

            _data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                    # transforms.Normalize((0.4676,0.3803, 0.3329), (0.2882, 0.2548, 0.2516)), # face
                ]),
                'val': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4676,0.3803, 0.3329), (0.2882, 0.2548, 0.2516)) # face
                    # transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]),
            }
            #sub



            self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                                    _data_transforms['train'])
            self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                                   _data_transforms['val'])
            logger.info('reading data done')
        elif self.params['type'] == config.TYPE_SUBIMAGENET:

            _data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [
                                         0.2302, 0.2265, 0.2262]),
                ]),
                'val': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [
                                         0.2302, 0.2265, 0.2262]),
                ]),
            }
            _data_dir = './data/sub-imagenet-200/'
            self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                                    _data_transforms['train'])
            self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                                   _data_transforms['val'])
            logger.info('reading data done')


        elif self.params['type'] == config.TYPE_WEBFACE:

            _data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]),
            }
            _data_dir = './data/tiny-imagenet-200/'
            self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                                    _data_transforms['train'])
            self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                                   _data_transforms['val'])
            logger.info('reading data done')


        self.classes_dict = self.build_classes_dict()
        print('self.classes_dict')
        print(len(self.classes_dict))
        logger.info('build_classes_dict done')
        if self.params['sampling_dirichlet']:
            ## sample indices for participants using Dirichlet distribution
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'], #100
                alpha=self.params['dirichlet_alpha'])
            # print(indices_per_participant)
            for pos, indices in indices_per_participant.items():
                print(f'Client {pos} dataset size: {len(indices)}')
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
            # print(111111)
            # print(train_loaders)
            if self.params['choice'] == 'gtrsb':
                
                seed2 = 2021
                hetero_dir_part = GTRSBPartitioner(self.train_dataset.labels, 
                                self.params['number_of_total_participants'],
                                balance=None, 
                                partition="dirichlet",
                                dir_alpha=self.params['dirichlet_alpha'],
                                seed=seed2)
 
 
                for pos, indices in hetero_dir_part.client_dict.items():
                    print(f'Client {pos} dataset size: {len(indices)}')
                train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             hetero_dir_part.client_dict.items()]

        else:
            ## sample indices for participants that are equally
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]

        logger.info('train loaders done')
        self.train_data = train_loaders
        # for pos in range(50):
        #     print(f'Client {pos} dataset size: {len(train_loaders[pos])}')
        self.test_data = self.get_test()
        self.test_data_poison ,self.test_targetlabel_data = self.poison_test_dataset()

        self.advasarial_namelist = self.params['adversary_list']

        if self.params['is_random_namelist'] == False:
            self.participants_list = self.params['participants_namelist']
        else:
            self.participants_list = list(range(self.params['number_of_total_participants']))
        # random.shuffle(self.participants_list)
        self.benign_namelist =list(set(self.participants_list) - set(self.advasarial_namelist))

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices),pin_memory=True, num_workers=8)
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               sub_indices))
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=False)
        return test_loader


    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target


    def maketrigger_iamge(self, bptt, evaluation=False, device = torch.device('cpu')) :

        device = torch.device('cpu')
        input_height = self.params['input_height'] # 224 28
        s = self.params['Poison_s'] # 0.4
        k = self.params['Poison_k'] # 50
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.interpolate(ins, size=input_height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
            .to(device)
        ) # 
        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...].to(device)
        
        grid_rescale = 1
        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)


        if evaluation:
            inputs, targets = bptt 
            bs = inputs.shape[0]
            inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            if self.params['use_fft']:
                inputs_bd = inputs
            inputs_bd2 = self.create_bd(inputs_bd, device)
            # inputs_bd2 = inputs_bd
            # targets = torch.ones_like(targets) * self.params['poison_label_swap']
            if self.params['blend']:
                transformed_inputs = []
                for input_image in inputs:
                    img_res = self.make_blend_image(input_image, device)
                    transformed_inputs.append(img_res)
                # print(type(transformed_inputs[0]))
                inputs_bd2 = torch.stack(transformed_inputs)
            targets = torch.ones_like(targets) * self.params['poison_label_swap']
            
            inputs_bd2.requires_grad_(False)
            targets.requires_grad_(False)
            if self.params['only_use_wanet']:
                inputs_bd2 = inputs_bd
                inputs_bd2.requires_grad_(False)
            return inputs_bd2, targets, bs
        else:
            poison_count = 0
            inputs, targets = bptt
            inputs.to(device)
            targets.to(device)
            poisoning_per_batch = self.params['poisoning_per_batch']
            rate_bd = poisoning_per_batch / 128
            rate_bd = 0.1
            bs = inputs.shape[0]
            cross_ratio = 0.1
            num_bd = int(math.ceil(bs * rate_bd)) 
            
            if num_bd <= 1:
                num_bd = 1
            num_cross = int(num_bd * cross_ratio)
            grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(num_cross, input_height, input_height, 2).to(device) * 2 - 1
            grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)
            inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
            # print(inputs_bd.size())
            if self.params['use_fft']:
                inputs_bd = inputs[:num_bd]
            inputs_bd2 = self.create_bd(inputs_bd, device)
            # inputs_bd2 = inputs_bd
            inputs_cross = F.grid_sample(inputs[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)

            if self.params['blend']:
                transformed_inputs = []
                                    # train_bd_transform = blend_attack_trans()
                for input_image in inputs[:num_bd]:
                    img_res = self.make_blend_image(input_image, device)
                    transformed_inputs.append(img_res)

                inputs_bd2 = torch.stack(transformed_inputs)

            if self.params['only_use_wanet']:
                inputs_bd2 = inputs_bd
            total_inputs = torch.cat([inputs_bd2, inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
            targets_bd = torch.ones_like(targets[:num_bd]) * self.params['poison_label_swap']
            total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
            poison_count = num_bd
            return total_inputs,total_targets,poison_count
    
    def create_bd(self, inputs, device ='cuda:0'):
        # device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        device = device
        device_cpu = torch.device("cpu")
        Poison_β = self.params['Poison_β'] 
        Poison_α = self.params['Poison_α'] 
        # datachoice = 'tiny-image'
        # datachoice = 'cifar'
        datachoice = self.params['environment_name']

        
        input_height = self.params['input_height'] # 32
        input_width = self.params['input_height']  # 32

        bs,_ ,_ ,_ = inputs.shape

        transforms_list = []
        transforms_list.append(transforms.Resize((input_height, input_width)))
        transforms_list.append(transforms.ToTensor())  
        transforms_class = transforms.Compose(transforms_list)

        if datachoice == 'sub-image' or datachoice == 'tiny' or datachoice == 'web-face':
            im_target = Image.open('./12.JPEG').convert('RGB')
            # im_target = Image.open('./13.JPEG').convert('RGB')
            im_target = Image.open('./14.JPEG').convert('RGB')
        elif datachoice == 'cifar':
            im_target = Image.open('./cifar_deer.JPEG').convert('RGB')
            im_target = Image.open('./gtsr.JPEG').convert('RGB')
            # im_target = Image.open('./minist-8.JPEG').convert('L')
            # im_target = Image.open('./cifar_deer.JPEG').convert('RGB')
        im_target = transforms_class(im_target)

        im_target = np.clip(im_target.numpy() * 255, 0, 255)
        im_target = torch.from_numpy(im_target).repeat(bs,1,1,1)

        # inputs = np.clip(inputs.numpy()*255,0,255)
        inputs_cpu = inputs.cpu()
        inputs_np = np.clip(inputs_cpu.numpy() * 255, 0, 255)

        bd_inputs = self.Fourier_pattern(inputs_np, im_target, Poison_β, Poison_α)
 

        bd_inputs = torch.tensor(np.clip(bd_inputs/255,0,1),dtype=torch.float32)

        return bd_inputs.to(device)


    def Fourier_pattern(self, img_, target_img, beta, ratio):
        img_=cp.asarray(img_)
        target_img=cp.asarray(target_img)
        #  get the amplitude and phase spectrum of trigger image
        fft_trg_cp = cp.fft.fft2(target_img, axes=(-2, -1))  
        amp_target, pha_target = cp.abs(fft_trg_cp), cp.angle(fft_trg_cp)  
        amp_target_shift = cp.fft.fftshift(amp_target, axes=(-2, -1))
        #  get the amplitude and phase spectrum of source image
        fft_source_cp = cp.fft.fft2(img_, axes=(-2, -1))
        amp_source, pha_source = cp.abs(fft_source_cp), cp.angle(fft_source_cp)
        amp_source_shift = cp.fft.fftshift(amp_source, axes=(-2, -1))

        # swap the amplitude part of local image with target amplitude spectrum
        bs,c, h, w = img_.shape
        b = (np.floor(np.amin((h, w)) * beta)).astype(int)  

        c_h = cp.floor(h / 2.0).astype(int)
        c_w = cp.floor(w / 2.0).astype(int)

        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1

        amp_source_shift[:,:, h1:h2, w1:w2] = amp_source_shift[:,:, h1:h2, w1:w2] * (1 - ratio) + (amp_target_shift[:,:,h1:h2, w1:w2]) * ratio
        # IFFT
        amp_source_shift = cp.fft.ifftshift(amp_source_shift, axes=(-2, -1))

        # get transformed image via inverse fft
        fft_local_ = amp_source_shift * cp.exp(1j * pha_source)
        local_in_trg = cp.fft.ifft2(fft_local_, axes=(-2, -1))
        local_in_trg = cp.real(local_in_trg)

        return cp.asnumpy(local_in_trg)



    def make_blend_image(self, input_image, device):
        img_t = plt.imread('./hello_kitty.jpeg') 
        img_t_pil = Image.fromarray(img_t)


        temp = self.params['input_height']
        img_t_resized_pil = img_t_pil.resize((temp, temp)) # 32 32 

        
        img_t = np.array(img_t_resized_pil)
        img_t = torch.tensor(img_t)
        img_t = img_t.permute(2, 0, 1).to(device)

        img_t = img_t.float()/255.0
        img_res = img_t*0.2 + input_image*0.8


        return img_res



    def get_poison_batch(self, bptt,adversarial_index=-1, evaluation=False):

        images, targets = bptt

        poison_count= 0
        new_images=images
        new_targets=targets

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = self.params['poison_label_swap']
                new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                # new_images[index] = self.maketrigger_iamge2(images[index], evaluation=True, device = torch.device('cpu')) 
                poison_count+=1

            else: # poison part of data when training
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['poison_label_swap']
                    new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                    # new_images[index] = self.maketrigger_iamge2(images[index], evaluation=True, device = torch.device('cpu')) 
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count

    def add_pixel_pattern(self,ori_image,adversarial_index):
        image = copy.deepcopy(ori_image)
        poison_patterns= []
        if adversarial_index==-1:
            for i in range(0,self.params['trigger_num']):
                poison_patterns = poison_patterns+ self.params[str(i) + '_poison_pattern']
        else :
            poison_patterns = self.params[str(adversarial_index) + '_poison_pattern']
        if self.params['type'] == config.TYPE_CIFAR or self.params['type'] == config.TYPE_TINYIMAGENET:
            for i in range(0,len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1
                image[1][pos[0]][pos[1]] = 1
                image[2][pos[0]][pos[1]] = 1


        elif self.params['type'] == config.TYPE_MNIST:

            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1

        return image

if __name__ == '__main__':
    np.random.seed(1)
    with open(f'./utils/cifar_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = ImageHelper(current_time=current_time, params=params_loaded,
                        name=params_loaded.get('name', 'mnist'))
    helper.load_data()

    pars= list(range(100))
    # show the data distribution among all participants.
    count_all= 0
    for par in pars:
        cifar_class_count = dict()
        for i in range(10):
            cifar_class_count[i] = 0
        count=0
        _, data_iterator = helper.train_data[par]
        for batch_id, batch in enumerate(data_iterator):
            data, targets= batch
            for t in targets:
                cifar_class_count[t.item()]+=1
            count += len(targets)
        count_all+=count
        print(par, cifar_class_count,count,max(zip(cifar_class_count.values(), cifar_class_count.keys())))

    print('avg', count_all*1.0/100)

def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)
    
class GTSRB(data.Dataset):
    def __init__(self, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join("./data", "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join("./data", "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label

class GTRSBPartitioner(CIFAR10Partitioner):
    """CIFAR100 data partitioner.

    This is a subclass of the :class:`CIFAR10Partitioner`. For details, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
    """
    num_classes = 43