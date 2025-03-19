import argparse
import collections

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data_loader.data_loaders as module_data
import models.model as module_arch
from utils.parse_config import ConfigParser


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


def label_to_one_hot(label, logit, ignore_index=255):
    target = torch.zeros_like(logit, device='cuda').float()
    for cls_idx in label.unique():
        if cls_idx in [0, ignore_index]:
            continue
        target[:, int(cls_idx) - 1] = (label == int(cls_idx)).float()
    return target


def main(config):
    step = config['data_loader']['args']['task']['step']
    n_epoch = config['trainer']['epochs']
    save_dir = str(config.save_dir)
    n_base_classes = int(config['data_loader']['args']['task']['name'].split('-')[0])
    n_inc_classes = int(config['data_loader']['args']['task']['name'].split('-')[1])
    n_seen_classes = n_base_classes + (step - 1) * n_inc_classes

    model_path = "{save_dir}{step}/checkpoint-epoch{n_epoch}.pth".format(
        save_dir=save_dir[:-len(save_dir.split('_')[-1])], step=step - 1, n_epoch=n_epoch)
    saved_info_path = "{save_dir}{step}/saved_info-epoch{n_epoch}.pth".format(
        save_dir=save_dir[:-len(save_dir.split('_')[-1])], step=step - 1, n_epoch=n_epoch)
    new_saved_info_path = "{save_dir}{step}/new_saved_info-epoch{n_epoch}.pth".format(
        save_dir=save_dir[:-len(save_dir.split('_')[-1])], step=step - 1, n_epoch=n_epoch)

    # for initializing prototypes
    dataset = config.init_obj(
        'data_loader',
        module_data
    )
    train_loader = dataset.get_train_loader(None)
    config_for_model = config
    config_for_model['data_loader']['args']['task']['step'] -= 1
    model = config_for_model.init_obj('arch', module_arch, **{"classes": dataset.get_per_task_classes()[:-1]}).cuda()
    model._load_pretrained_model(f'{model_path}')
    model.eval()

    prototypes = torch.zeros(n_seen_classes, 256, device='cuda')
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):

            if batch_idx % (len(train_loader) // 5) == 0:
                print('[' + str(batch_idx) + '/' + str(len(train_loader)) + ']')

            logit, features, _ = model(data['image'].cuda(), ret_intermediate=True)

            pred = logit.argmax(dim=1) + 1
            idx = (logit > 0.5).float()
            idx = idx.sum(dim=1)
            pred[idx == 0] = 0

            target = label_to_one_hot(pred, logit)

            small_target = target[:, :, 8::16, 8::16]
            normalized_features = F.normalize(features[-1], p=2, dim=1)

            class_region = small_target.unsqueeze(2) * normalized_features.unsqueeze(1)
            prototypes = prototypes + class_region.sum(dim=[0, 3, 4])
    prototypes = F.normalize(prototypes, p=2, dim=1)

    # for computing prototypes
    prev_model = torch.load(model_path)
    saved_info = torch.load(saved_info_path)
    if step == 1:
        cls_weight = prev_model['state_dict']['cls.0.weight'].squeeze(2).squeeze(2)
        cls_bias = prev_model['state_dict']['cls.0.bias']
    else:
        cls_weight = []
        cls_bias = []
        for i in range(step):
            cls_weight.append(prev_model['state_dict']['cls.{}.weight'.format(i)].squeeze(2).squeeze(2))
            cls_bias.append(prev_model['state_dict']['cls.{}.bias'.format(i)])
        cls_weight = torch.cat(cls_weight, dim=0)
        cls_bias = torch.cat(cls_bias, dim=0)

    input_size = cls_weight.shape[1]
    num_classes = cls_weight.shape[0]
    model = MLP(input_size, num_classes)
    model.fc.weight.data = cls_weight
    model.fc.bias.data = cls_bias

    # initialize prototypes
    x = torch.zeros(num_classes, input_size, requires_grad=True, device=cls_weight.device)

    target = torch.zeros(num_classes, num_classes, device=cls_weight.device)
    for i in range(num_classes):
        target[i, i] = 1

    optimizer = optim.AdamW([x], lr=0.00025, weight_decay=0.01)

    # compute prototypes
    best_loss = 100000000
    n_iters = 500000
    for step in range(n_iters):
        optimizer.zero_grad()

        logits = model(x)
        pos_weight = torch.ones(num_classes, device=cls_weight.device)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(0)

        if loss.item() < best_loss:
            best_loss = loss.item()
            computed_prototype = F.normalize(x, dim=1)

        if step % 5000 == 0:
            print(f"Step {step}: loss={loss.item()}",
                  f"Step {step}: norm={torch.norm(x, dim=1).mean()}")

    # ultimate prototypes
    computed_prototype = F.normalize(computed_prototype + prototypes, dim=1)

    # compute angles
    if config['trainer']['noise_type'] == 'angle':
        cos_sim = (computed_prototype.unsqueeze(0) * computed_prototype.unsqueeze(1)).sum(dim=2)
        all_angles = torch.acos(cos_sim.clamp(-1, 1))

        computed_angles = []
        for cls in range(num_classes):
            computed_angle = torch.cat([all_angles[cls, :cls], all_angles[cls, cls + 1:]]).min()
            computed_angles.append(computed_angle / 2)
        computed_angles = torch.stack(computed_angles)

        computed_prototype.detach_()
        computed_angles.detach_()
        new_saved_info = {
            'prototypes': computed_prototype,
            'angle_noise': computed_angles,
            'numbers': saved_info['numbers']
        }
    elif config['trainer']['noise_type'] == 'dist':
        all_dists = torch.norm(computed_prototype.unsqueeze(0) - computed_prototype.unsqueeze(1), dim=2)
        computed_dists = []
        for cls in range(num_classes):
            computed_dist = torch.cat([all_dists[cls, :cls], all_dists[cls, cls + 1:]]).min()
            computed_dists.append(computed_dist / 1)
        computed_dists = torch.stack(computed_dists)

        computed_prototype.detach_()
        computed_dists.detach_()
        new_saved_info = {
            'prototypes': computed_prototype,
            'dist_noise': computed_dists,
            'numbers': saved_info['numbers']
        }
    elif config['trainer']['noise_type'] == 'bitwise':
        raise NotImplementedError

    torch.save(new_saved_info, new_saved_info_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Class incremental Semantic Segmentation')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type action target', defaults=(None, float, None, None))
    options = [
        CustomArgs(['--multiprocessing_distributed'], action='store_true', target='multiprocessing_distributed'),
        CustomArgs(['--dist_url'], type=str, target='dist_url'),

        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--save_dir'], type=str, target='trainer;save_dir'),

        CustomArgs(['--mem_size'], type=int, target='data_loader;args;memory;mem_size'),

        CustomArgs(['--seed'], type=int, target='seed'),
        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;train;batch_size'),

        CustomArgs(['--task_name'], type=str, target='data_loader;args;task;name'),
        CustomArgs(['--task_step'], type=int, target='data_loader;args;task;step'),
        CustomArgs(['--task_setting'], type=str, target='data_loader;args;task;setting'),

        CustomArgs(['--pos_weight'], type=float, target='hyperparameter;pos_weight'),
        CustomArgs(['--mbce'], type=float, target='hyperparameter;mbce'),

        CustomArgs(['--freeze_bn'], action='store_true', target='arch;args;freeze_all_bn'),
        CustomArgs(['--test'], action='store_true', target='test'),

        CustomArgs(['--noise_type'], type=str, target='trainer;noise_type'),
        CustomArgs(['--basemodel'], type=str, target='arch;type'),
    ]
    config = ConfigParser.from_args(args, options)

    torch.distributed.init_process_group(
        backend=config['dist_backend'], init_method=config['dist_url'],
        world_size=config['world_size'], rank=config['rank']
    )

    main(config)
