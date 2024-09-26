import utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import main
import test
import copy
import config
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cupy as cp
from collections import OrderedDict



def ImageTrain(helper, start_epoch, local_model, target_model, is_poison,agent_name_keys):


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_cpu = torch.device("cpu")
    epochs_submit_update_dict = dict()
    epochs_submit_update_dict_order = dict()
    num_samples_dict = dict()
    list_res = []
    train_models = []

    client_models = []

    for ii in range(helper.params['no_models']):
        # print(ii)
        client_models.append(copy.deepcopy(target_model))
    current_number_of_adversaries=0
    for temp_name in agent_name_keys:
        if temp_name in helper.params['adversary_list']:
            current_number_of_adversaries+=1

    for model_id in range(helper.params['no_models']):
        epochs_local_update_list = []
        epochs_local_update_list_order = []
        last_local_model = dict()
        client_grad = [] # only works for aggr_epoch_interval=1

        for name, data in target_model.state_dict().items():
            last_local_model[name] = target_model.state_dict()[name].clone()

        agent_name_key = agent_name_keys[model_id]
        ## Synchronize LR and models
        # model = local_model
        # model.copy_params(target_model.state_dict())

        temp_model = copy.deepcopy(target_model)
        model = temp_model
        # model = target_model
        # for key, value in model.state_dict().items():
        #     target_value  = target_model.to(value.device)
        #     # new_value = target_value + (value - target_value) * 1

            
        #     model.state_dict()[key].copy_(target_value)
        # def copy_params(model, target_params_variables):
        # for name, layer in model.named_parameters():
        #     layer.data = copy.deepcopy(target_params_variables[name])
        # lr_1 = 0.001
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        lr = optimizer.param_groups[0]['lr']
        # print("LR ：", lr)
        #debug norm
        target_model_copy=dict()
        localmodel_poison_epochs = helper.params['poison_epochs']
        
        ###
        for name, param in target_model.named_parameters():
            target_model_copy[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)
        #debug end


        model.train()
        adversarial_index= -1
        localmodel_poison_epochs = helper.params['poison_epochs']
        if is_poison and agent_name_key in helper.params['adversary_list']:
            for temp_index in range(0, len(helper.params['adversary_list'])):
                if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
                    adversarial_index= temp_index
                    localmodel_poison_epochs = helper.params[str(temp_index) + '_poison_epochs']
                    main.logger.info(
                        f'poison local model {agent_name_key} index {adversarial_index} ')
                    break
            if len(helper.params['adversary_list']) == 1:
                adversarial_index = -1  # the global pattern

        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):

            target_params_variables = dict()
            for name, param in target_model.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            if is_poison and agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                main.logger.info('poison_now')

                poison_lr = helper.params['poison_lr']
                internal_epoch_num = helper.params['internal_poison_epochs']
                step_lr = helper.params['poison_step_lr']

                poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                                   momentum=helper.params['momentum'],
                                                   weight_decay=helper.params['decay'])
                scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                 milestones=[0.2 * internal_epoch_num,
                                                                             0.8 * internal_epoch_num], gamma=0.1)
                # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.1)
                temp_local_epoch = (epoch - 1) *internal_epoch_num


                #debug

                rate_bd = 0.1 #
                # rate_bd = 0.01
                total_loss_ce = 0
                total_sample = 0
                cross_ratio = 0.1
                s = 0.5
                grid_rescale = 1
                input_height = 32
                    
                k = 4
                target_label = 2
                # Prepare grid
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


                for internal_epoch in range(1, internal_epoch_num + 1):
                    temp_local_epoch += 1
                    _, data_iterator = helper.train_data[agent_name_key]
                    poison_data_count = 0
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list=[]

                    ###
                    model.train()
                    # model.to(device)
                    for batch_id, batch in enumerate(data_iterator):
                        # device = torch.device('cpu')
                        poison_optimizer.zero_grad()
                        data, targets = batch
                        dataset_size += len(data)
                        inputs, targets = data.to(device), targets.to(device)
                        bs = inputs.shape[0]
                        
                        data, targets, poison_num = helper.maketrigger_iamge(batch, False, device)
                        # data, targets = batch
                        # # # 
                        data = data.cuda()
                        targets = targets.cuda()
                        # #CBA  DBA 
                        # data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adversarial_index,evaluation=False)
                        # data, targets = batch
                        # data = data.cuda()
                        # targets = targets.cuda()

                        # poison_optimizer.zero_grad()
                        
                        poison_data_count += poison_num
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)

                        # distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                        # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
                        # loss = helper.params['alpha_loss'] * class_loss + \
                        #        (1 - helper.params['alpha_loss']) * distance_loss
                        loss.backward()

                        epsilon = 0.01
                        weight_difference, difference_flat = get_weight_difference(target_model_copy, model.named_parameters())
                        clipped_weight_difference, _ = clip_grad(epsilon, weight_difference, difference_flat) # 0.3

                        # weight_difference, difference_flat = get_weight_difference(target_model_copy, clipped_weight_difference)
                        # copy_params(model, weight_difference)
                        # # main.logger.info(f'mine')
                        # for name, layer in model.named_parameters():
                        #     # 使用全局权重和裁剪后的权重差异进行更新
                        #     layer.data = target_model_copy[name].data + clipped_weight_difference[name]
                        # # for (name, param), diff in zip(model.named_parameters(), weight_difference.values()):
                        #     param.data += diff
                        # copy_params(model, weight_difference)
                        # get gradients
                        # 更新模型权重
                        # main.logger.info(f'gamma = 1')
                        for name, layer in model.named_parameters():
                            # 计算 L_j^(t+1) - G^t
                            L_j_t_plus_1 = layer.data
                            G_t = target_model_copy[name].data
                            # 使用公式进行归一化
                            norm_difference = torch.norm(L_j_t_plus_1 - G_t)
                            gamma = 1 # 设定 gamma 的值  # gtr 1
                            
                            normalization_factor = max(1, (norm_difference / epsilon) + (gamma / ((norm_difference + gamma) * epsilon)))
                            # 更新权重
                            layer.data = G_t + (clipped_weight_difference[name] / normalization_factor)



                        if helper.params['aggregation_methods']==config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        poison_optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                        if helper.params["batch_track_distance"]:
                            # we can calculate distance to this model now.
                            temp_data_len = len(data_iterator)
                            distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                            dis2global_list.append(distance_to_global_model)
                            model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                           data_len=temp_data_len,
                                                            batch=batch_id,distance_to_global_model= distance_to_global_model,
                                                           eid=helper.params['environment_name'],
                                                           name=str(agent_name_key),is_poisoned=True)

                    if step_lr:
                        scheduler.step()
                        main.logger.info(f'Current lr: {scheduler.get_lr()}')

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    main.logger.info(
                        ',  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}'.format( epoch, agent_name_key, # model.name
                                                                                      internal_epoch,
                                                                                      total_l, correct, dataset_size,
                                                                                     acc, poison_data_count))
                    csv_record.train_result.append(
                        [agent_name_key, temp_local_epoch,
                         epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])
                    if helper.params['vis_train']:
                        model.train_vis(main.vis, temp_local_epoch,
                                        acc, loss=total_l, eid=helper.params['environment_name'], is_poisoned=True,
                                        name=str(agent_name_key) )
                    num_samples_dict[agent_name_key] = dataset_size
                    if helper.params["batch_track_distance"]:
                        main.logger.info(
                            f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                            f'Distance to the global model: {dis2global_list}. ')

                # internal epoch finish
                main.logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
                main.logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                                 f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

                if not helper.params['baseline']:
                    main.logger.info(f'will scale.')
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                                   model=model, is_poison=False,
                                                                                   visualize=False,
                                                                                   agent_name_key=agent_name_key)
                    csv_record.test_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          is_poison=True,
                                                                                          visualize=False,
                                                                                          agent_name_key=agent_name_key)
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    clip_rate = helper.params['scale_weights_poison']
                    main.logger.info(f"Scaling by  {clip_rate}")
                    for key, value in model.state_dict().items():
                        target_value  = last_local_model[key]
                        new_value = target_value + (value - target_value) * clip_rate
                        model.state_dict()[key].copy_(new_value)
                    distance = helper.model_dist_norm(model, target_params_variables)
                    main.logger.info(
                        f'Scaled Norm after poisoning: '
                        f'{helper.model_global_norm(model)}, distance: {distance}')
                    csv_record.scale_temp_one_row.append(epoch)
                    csv_record.scale_temp_one_row.append(round(distance, 4))
                    if helper.params["batch_track_distance"]:
                        temp_data_len = len(helper.train_data[agent_name_key][1])
                        model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                       data_len=temp_data_len,
                                                       batch=temp_data_len-1,
                                                       distance_to_global_model=distance,
                                                       eid=helper.params['environment_name'],
                                                       name=str(agent_name_key), is_poisoned=True)

                distance = helper.model_dist_norm(model, target_params_variables)
                main.logger.info(f"Total norm for {current_number_of_adversaries} "
                                 f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")
                

                # weight_difference, difference_flat = get_weight_difference(target_model_copy, model.named_parameters())
                # clipped_weight_difference, _ = clip_grad(0.03, weight_difference, difference_flat) # 0.3
                # weight_difference, difference_flat = get_weight_difference(target_model_copy, clipped_weight_difference)
                # copy_params(model, weight_difference)

            else:
                temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']
                for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                    temp_local_epoch += 1

                    _, data_iterator = helper.train_data[agent_name_key]
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list = []
                    for batch_id, batch in enumerate(data_iterator):
                        
                        optimizer.zero_grad()
                        data, targets = helper.get_batch(data_iterator, batch,evaluation=False)

                        dataset_size += len(data)
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)
                        loss.backward()

                        # get gradients
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                        if helper.params["vis_train_batch_loss"]:
                            cur_loss = loss.data
                            temp_data_len = len(data_iterator)
                            model.train_batch_vis(vis=main.vis,
                                                  epoch=temp_local_epoch,
                                                  data_len=temp_data_len,
                                                  batch=batch_id,
                                                  loss=cur_loss,
                                                  eid=helper.params['environment_name'],
                                                  name=str(agent_name_key) , win='train_batch_loss', is_poisoned=False)
                        if helper.params["batch_track_distance"]:
                            # we can calculate distance to this model now
                            temp_data_len = len(data_iterator)
                            distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                            dis2global_list.append(distance_to_global_model)
                            model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                           data_len=temp_data_len,
                                                            batch=batch_id,distance_to_global_model= distance_to_global_model,
                                                           eid=helper.params['environment_name'],
                                                           name=str(agent_name_key),is_poisoned=False)

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    main.logger.info(
                        ' epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format( epoch, agent_name_key, internal_epoch, # model.name,
                                                           total_l, correct, dataset_size,
                                                           acc))
                    csv_record.train_result.append([agent_name_key, temp_local_epoch,
                                                    epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

                    if helper.params['vis_train']:
                        model.train_vis(main.vis, temp_local_epoch,
                                        acc, loss=total_l, eid=helper.params['environment_name'], is_poisoned=False,
                                        name=str(agent_name_key))
                    num_samples_dict[agent_name_key] = dataset_size

                    if helper.params["batch_track_distance"]:
                        main.logger.info(
                            f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                            f'Distance to the global model: {dis2global_list}. ')

                # test local model after internal epoch finishing
                # epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                #                                                                model=model, is_poison=False, visualize=False,
                #                                                                agent_name_key=agent_name_key)
                # csv_record.test_result.append([agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

            if is_poison:
                if agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          is_poison=True,
                                                                                          visualize=False,
                                                                                          agent_name_key=agent_name_key)
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                #  test on local triggers
                if agent_name_key in helper.params['adversary_list']:
                    if helper.params['vis_trigger_split_test']:
                        model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                     eid=helper.params['environment_name'],
                                                     name=str(agent_name_key)  + "_combine")

                    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
                        test.Mytest_poison_agent_trigger(helper=helper, model=model, agent_name_key=agent_name_key)
                    # csv_record.poisontriggertest_result.append(
                    #     [agent_name_key, str(agent_name_key) + "_trigger", "", epoch, epoch_loss,
                    #      epoch_acc, epoch_corret, epoch_total])
                    if helper.params['vis_trigger_split_test']:
                        model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                     eid=helper.params['environment_name'],
                                                     name=str(agent_name_key) + "_trigger")
           
            # 初始化本地更新字典
            local_model_update_dict_order = OrderedDict() 

            for name, data in model.state_dict().items():
                # 其他记录更新量的代码
                # 将name作为key加入OrderedDict
                local_model_update_dict_order[name] = torch.zeros_like(data)
                local_model_update_dict_order[name] = (data - last_local_model[name])
            
            # update the model weight
            local_model_update_dict = dict()
            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros_like(data)
                local_model_update_dict[name] = (data - last_local_model[name])
                last_local_model[name] = copy.deepcopy(data)

            # train_models.append(model)



            if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                epochs_local_update_list.append(client_grad)
            else:
                epochs_local_update_list.append(local_model_update_dict)
                epochs_local_update_list_order.append(local_model_update_dict_order)

        epochs_submit_update_dict[agent_name_key] = epochs_local_update_list
        epochs_submit_update_dict_order[agent_name_key] = epochs_local_update_list_order
        list_res.append(epochs_local_update_list_order)
        train_models.append(model)
        del model
             # 释放 GPU 缓存
        torch.cuda.empty_cache()
        # epochs_submit_update_dict[agent_name_key] = local_model_update_dict
    
    ordered_dict = OrderedDict.fromkeys(epochs_submit_update_dict)
    ordered_dict2 = OrderedDict()
    ordered_dict2.update(epochs_submit_update_dict)
    # for k, v in ordered_dict.items():
    #     if epochs_submit_update_dict_order[k] != v:
    #         print(1)
    #     else:
    #         print(2)
    # ordered_dict = OrderedDict((k, epochs_submit_update_dict[k]) for k in sorted(epochs_submit_update_dict))
    # for k, v in ordered_dict.items():
    #     if epochs_submit_update_dict_order[k] != v:
    #         print(1)
    #     else:
    #         print(2)
    print(len(train_models))
    ordered_dicts = [dct for v in epochs_submit_update_dict_order.values() for dct in v]
    list1 = []
    for key in num_samples_dict.keys():
        list1.append(num_samples_dict[key])
        # list2.append(epochs_submit_update_dict_order(key))
    merged2 = list(zip(list1, ordered_dicts))
    local_state_dicts = [model.state_dict() for model in train_models]
    dict1, dict2 = local_state_dicts[0], local_state_dicts[1] 
    # for key in dict1:
    #     if not torch.equal(dict1[key], dict2[key]):
    #         print("Models are different!")
    #     else:
    #         print(1111111)
    merged3 = list(zip(list1, local_state_dicts))
    # print(merged3)
    # local_state_dicts = [i for i in epochs_submit_update_dict_order]
    # # for key in epochs_submit_update_dict_order:
    # #     list2.append(epochs_submit_update_dict_order(key))
    # merged = list(zip(list1,local_state_dicts))
    return epochs_submit_update_dict, num_samples_dict, merged3


def get_weight_difference(weight1, weight2):
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)

        return difference, difference_flat
    
    # @staticmethod
def clip_grad(norm_bound, weight_difference, difference_flat):

    l2_norm = torch.norm(difference_flat.clone().detach().cuda())
    
    scale =  max(1.0, float(torch.abs(l2_norm / norm_bound)))
    # print("scale: ", scale)
    for name in weight_difference.keys():
        weight_difference[name].div_(scale)

    return weight_difference, l2_norm
    
def copy_params(model, target_params_variables):
    for name, layer in model.named_parameters():
        layer.data = copy.deepcopy(target_params_variables[name])


def make_blend_image(input_image, device):
    # print(type(input_image))


    img_t = plt.imread('/home/changrx/last/BackdoorBench-main/resource/blended/hello_kitty.jpeg') 
    img_t_pil = Image.fromarray(img_t)

    # 调整图像大小为32x32像素
    # img_t_resized_pil = img_t_pil.resize((32, 32))
    img_t_resized_pil = img_t_pil.resize((32, 32))

    # 如果你需要将调整大小后的PIL图像再次转换为NumPy数组：
    img_t = np.array(img_t_resized_pil)
    img_t = torch.tensor(img_t)
    img_t = img_t.permute(2, 0, 1).to(device)
    # print(img_t.size())
    # print(input_image.size())
    img_t = img_t.float()/255.0
    img_res = img_t*0.2 + input_image*0.8
    # print(type(img_res))

    return img_res
    # transformed_inputs.append(img_res)

def maketrigger_iamge( bptt, device = torch.device('cpu')) :
        device = torch.device('cpu')
        poison_count = 0
        inputs, targets = bptt
        inputs.to(device)
        targets.to(device)
        rate_bd = 0.1
        bs = inputs.shape[0]
        cross_ratio = 0.1
        num_bd = int(bs * rate_bd) # 中毒的图片数量
        input_height = 32
        s = 0.15
        k = 4
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



        if num_bd <= 1:
            num_bd = 1
        num_cross = int(num_bd * cross_ratio)
        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(num_cross, input_height, input_height, 2).to(device) * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)
        inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        inputs_bd2 = create_bd(inputs_bd, device)
        inputs_cross = F.grid_sample(inputs[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)


        transformed_inputs = []
                            # train_bd_transform = blend_attack_trans()
        # for input_image in inputs[:num_bd]:
        #     # 应用train_bd_transform到每个图像
        #     img_res = self.make_blend_image(input_image, device)
        #     transformed_inputs.append(img_res)
        # # print(type(transformed_inputs[0]))
        # inputs_bd2 = torch.stack(transformed_inputs)





        total_inputs = torch.cat([inputs_bd2, inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
        targets_bd = torch.ones_like(targets[:num_bd]) * self.params['poison_label_swap']
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        poison_count = num_bd
        return total_inputs,total_targets,poison_count
 
    
def create_bd( inputs, device ='cuda:0'):
        # device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    device = device
    device_cpu = torch.device("cpu")
    Poison_β = 0.1 # 0.05
    Poison_α = 0.15 #0.05
    datachoice = 'cifar'
    # if datachoice == 'cifar':
    #     input_height = 32
    #     input_width = 32
    # elif datachoice == 'sub-image' or datachoice == 'tiny-image' or datachoice == 'web-face':
    #     input_height = 224
    #     input_width = 224
    
    input_height = 32
    input_width = 32
    bs,_ ,_ ,_ = inputs.shape

    transforms_list = []
    transforms_list.append(transforms.Resize((input_height, input_width)))
    transforms_list.append(transforms.ToTensor())  
    transforms_class = transforms.Compose(transforms_list)

    if datachoice == 'sub-image' or datachoice == 'tiny-image' or datachoice == 'web-face':
        im_target = Image.open('/home/changrx/MachineCode/silent-killer-main/imgs/12.JPEG').convert('RGB')
        # im_target = Image.open('/home/changrx/MachineCode/silent-killer-main/imgs/13.JPEG').convert('RGB')
        im_target = Image.open('/home/changrx/MachineCode/silent-killer-main/imgs/14.JPEG').convert('RGB')
    elif datachoice == 'cifar':
        im_target = Image.open('/home/changrx/MachineCode/silent-killer-main/imgs/cifar_deer.JPEG').convert('RGB')
        # im_target = Image.open('/home/changrx/MachineCode/silent-killer-main/cifar100_test/data/0.png').convert('RGB')
        # im_target = Image.open('/home/changrx/MachineCode/silent-killer-main/cifar100_test/data/1.png').convert('RGB')
        # im_target = Image.open('/home/changrx/MachineCode/silent-killer-main/imgs/cifar_deer.JPEG').convert('RGB')
    im_target = transforms_class(im_target)

    im_target = np.clip(im_target.numpy() * 255, 0, 255)
    im_target = torch.from_numpy(im_target).repeat(bs,1,1,1)

    # inputs = np.clip(inputs.numpy()*255,0,255)
    # 将 inputs 张量从 GPU 复制到 CPU
    inputs_cpu = inputs.cpu()

    # 将 inputs_cpu 转换为 NumPy 数组，并进行数值裁剪
    inputs_np = np.clip(inputs_cpu.numpy() * 255, 0, 255)

    bd_inputs = Fourier_pattern(inputs_np, im_target, Poison_β, Poison_α)

    bd_inputs = torch.tensor(np.clip(bd_inputs/255,0,1),dtype=torch.float32)


    return bd_inputs.to(device)


def Fourier_pattern( img_, target_img, beta, ratio):
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
        # 中心点
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

