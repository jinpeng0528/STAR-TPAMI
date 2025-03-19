import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter



class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config, logger, gpu):
        self.config = config
        
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['epochs'] if cfg_trainer['save_period'] == -1 else cfg_trainer['save_period']
        # self.save_period = 1
        self.validation_period = cfg_trainer['validation_period'] if cfg_trainer['validation_period'] == -1 else cfg_trainer['validation_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.reset_best_mnt = cfg_trainer['reset_best_mnt']
        self.rank = torch.distributed.get_rank()

        # if config['data_loader']['args']['task']['step'] > 0:
        self.save_prototypes = cfg_trainer['save_prototypes']
        self.noise_type = cfg_trainer['noise_type']
        self.save_norms = cfg_trainer['save_norms']

        if logger is None:
            self.logger = config.get_logger('trainer', cfg_trainer['verbosity'])
        else:
            self.logger = logger
            # setup visualization writer instance
            if self.rank == 0:
                self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
            else:
                self.writer = TensorboardWriter(config.log_dir, self.logger, False)
        
        if gpu is None:
            # setup GPU device if available, move model into configured device
            self.device, self.device_ids = self._prepare_device(config['n_gpu'])
        else:
            self.device = gpu
            self.device_ids = None

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir


        # if config.resume is not None:
        #     self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, val_flag = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            
            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.rank == 0:
                if val_flag and (self.mnt_mode != 'off'):
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        self.logger.warning("Warning: Metric '{}' is not found. "
                                            "Model performance monitoring is disabled.".format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        self._save_best_model(epoch)
                        
                    else:
                        not_improved_count += 1

                    if (self.early_stop > 0) and (not_improved_count > self.early_stop):
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                         "Training stops.".format(self.early_stop))
                        break

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch)

                    if self.save_prototypes:
                        self.compute_prototypes(self.config)
                        assert self.noise_type in ['angle', 'bitwise', 'dist']
                        if self.noise_type == 'angle':
                            self.compute_angle_noise(self.config)
                        elif self.noise_type == 'bitwise':
                            self.compute_bitwise_noise(self.config)
                        elif self.noise_type == 'dist':
                            self.compute_dist_noise(self.config)

                    self.save_info(self.config, epoch)

        # close TensorboardX
        self.writer.close()

    def save_info(self, config, epoch):
        save_file = str(config.save_dir) + "/saved_info-epoch{}.pth".format(epoch)
        if self.save_prototypes:
            if self.noise_type == 'angle':
                all_info = {
                    "numbers": self.numbers,
                    "prototypes": self.prototypes,
                    "angle_noise": self.angle_noise,
                }
            elif self.noise_type == 'bitwise':
                all_info = {
                    "numbers": self.numbers,
                    "prototypes": self.prototypes,
                    "bitwise_noise": self.bitwise_noise,
                }
            elif self.noise_type == 'dist':
                all_info = {
                    "numbers": self.numbers,
                    "prototypes": self.prototypes,
                    "dist_noise": self.dist_noise,
                }
            # if self.save_norms:
            #     all_info["norm_mean_and_std"] = self.norm_mean_and_std
        else:
            all_info = {
                "numbers": self.numbers,
            }
        torch.save(all_info, save_file)

    def compute_cls_number(self, config):
        self.logger.info("computing number of pixels...")

        number_save_file = str(config.save_dir) + "/numbers_tmp.pth"
        if os.path.exists(number_save_file):
            self.numbers = torch.load(number_save_file)
            return

        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes

        numbers = torch.zeros(n_new_classes + 1).to(self.device)

        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):
                small_label = data['label'][:, 8::16, 8::16].to(self.device)
                for i in range(n_new_classes + 1):
                    if i == 0:
                        numbers[i] = numbers[i] + torch.sum(small_label == 0).item()
                        continue
                    numbers[i] = numbers[i] + torch.sum(small_label == i + n_old_classes).item()
                self.progress(self.logger, batch_idx, len(self.train_loader))

        self.numbers = numbers

        torch.save(numbers, number_save_file)

    def compute_prototypes(self, config):
        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes

        prototypes = torch.zeros(n_new_classes, 256, device='cuda')
        norms = {k: [] for k in range(n_new_classes)}
        # norm_mean_and_std = torch.zeros(2, n_new_classes, device='cuda')

        self.logger.info("computing prototypes...")
        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):

                logit, features, _ = self.model(data['image'].cuda(), ret_intermediate=True)

                target = label_to_one_hot(data['label'], logit[:, -n_new_classes:], n_old_classes)
                small_target = target[:, :, 8::16, 8::16]
                small_label = data['label'][:, 8::16, 8::16]

                normalized_features = F.normalize(features[-1], p=2, dim=1)
                class_region = small_target.unsqueeze(2) * normalized_features.unsqueeze(1)
                prototypes = prototypes + class_region.sum(dim=[0, 3, 4])

                norm = torch.norm(features[-1], p=2, dim=1)
                for cls in small_label.unique():
                    if cls in [0, 255]:
                        continue
                    norms[int(cls) - n_old_classes - 1].append(norm[small_label == cls])

                self.progress(self.logger, batch_idx, len(self.train_loader))

            prototypes = F.normalize(prototypes, p=2, dim=1)

            if config['data_loader']['args']['task']['step'] == 0:
                self.prototypes = prototypes
            else:
                self.prototypes = torch.cat([self.prev_prototypes, prototypes], dim=0)

            # for k in range(n_new_classes):
            #     norms[k] = torch.cat(norms[k], dim=0)
            #     norm_mean_and_std[0, k] = norms[k].mean()
            #     norm_mean_and_std[1, k] = norms[k].std()
            #
            # if config['data_loader']['args']['task']['step'] == 0 and self.save_norms:
            #     self.norm_mean_and_std = norm_mean_and_std
            # elif config['data_loader']['args']['task']['step'] > 0 and self.save_norms:
            #     self.norm_mean_and_std = torch.cat([self.prev_norm, norm_mean_and_std], dim=1)

    def compute_bitwise_noise(self, config):
        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes

        prototypes = self.prototypes

        bitwise_noise = torch.zeros(n_new_classes, 256, device='cuda')
        cnt = torch.zeros(n_new_classes, device='cuda')

        self.logger.info("computing bitwise_noise...")
        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):

                logit, features, _ = self.model(data['image'].cuda(), ret_intermediate=True)

                target = label_to_one_hot(data['label'], logit[:, -n_new_classes:], n_old_classes)
                small_target = target[:, :, 8::16, 8::16]
                small_label = data['label'][:, 8::16, 8::16]

                normalized_features = F.normalize(features[-1], p=2, dim=1)
                class_region = small_target.unsqueeze(2) * normalized_features.unsqueeze(1)

                for cls in small_label.unique():
                    if cls in [0, 255]:
                        continue
                    bitwise_diff = \
                        class_region[:, int(cls) - n_old_classes - 1].permute(1, 0, 2, 3)[:, small_label == cls] - \
                        prototypes[int(cls) - n_old_classes - 1].unsqueeze(1)
                    bitwise_diff = bitwise_diff ** 2

                    bitwise_noise[int(cls) - n_old_classes - 1] += (bitwise_diff ** 2).sum(dim=1)
                    cnt[int(cls) - n_old_classes - 1] = \
                        cnt[int(cls) - n_old_classes - 1] + (small_label == cls).sum()

                self.progress(self.logger, batch_idx, len(self.train_loader))

            bitwise_noise = torch.sqrt(bitwise_noise / cnt.unsqueeze(1))
            self.bitwise_noise = bitwise_noise

            if config['data_loader']['args']['task']['step'] == 0:
                self.bitwise_noise = bitwise_noise
            else:
                self.bitwise_noise = torch.cat([self.prev_bitwise_noise, bitwise_noise], dim=0)

    def compute_dist_noise(self, config):
        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes

        prototypes = self.prototypes

        dist_noise = torch.zeros(n_new_classes, device='cuda')
        cnt = torch.zeros(n_new_classes, device='cuda')

        self.logger.info("computing dist_noise...")
        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):

                logit, features, _ = self.model(data['image'].cuda(), ret_intermediate=True)

                target = label_to_one_hot(data['label'], logit[:, -n_new_classes:], n_old_classes)
                small_target = target[:, :, 8::16, 8::16]
                small_label = data['label'][:, 8::16, 8::16]

                normalized_features = F.normalize(features[-1], p=2, dim=1)
                class_region = small_target.unsqueeze(2) * normalized_features.unsqueeze(1)

                for cls in small_label.unique():
                    if cls in [0, 255]:
                        continue
                    bitwise_diff = \
                        class_region[:, int(cls) - n_old_classes - 1].permute(1, 0, 2, 3)[:, small_label == cls] - \
                        prototypes[int(cls) - n_old_classes - 1].unsqueeze(1)
                    dist_diff = torch.norm(bitwise_diff, p=2, dim=0)
                    dist_noise[int(cls) - n_old_classes - 1] += (dist_diff ** 2).sum()

                    cnt[int(cls) - n_old_classes - 1] = \
                        cnt[int(cls) - n_old_classes - 1] + (small_label == cls).sum()

                self.progress(self.logger, batch_idx, len(self.train_loader))

            dist_noise = torch.sqrt(dist_noise / cnt)
            self.dist_noise = dist_noise

            if config['data_loader']['args']['task']['step'] == 0:
                self.dist_noise = dist_noise
            else:
                self.dist_noise = torch.cat([self.prev_dist_noise, dist_noise], dim=0)

    def compute_angle_noise(self, config):
        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes

        prototypes = self.prototypes

        angle_noise = torch.zeros(n_new_classes, device='cuda')
        cnt = torch.zeros(n_new_classes, device='cuda')

        self.logger.info("computing angle_noise...")
        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):

                logit, features, _ = self.model(data['image'].cuda(), ret_intermediate=True)

                target = label_to_one_hot(data['label'], logit[:, -n_new_classes:], n_old_classes)
                small_target = target[:, :, 8::16, 8::16]
                small_label = data['label'][:, 8::16, 8::16]

                normalized_features = F.normalize(features[-1], p=2, dim=1)
                class_region = small_target.unsqueeze(2) * normalized_features.unsqueeze(1)

                for cls in small_label.unique():
                    if cls in [0, 255]:
                        continue
                    cos_similarity = torch.matmul(
                        class_region[:, int(cls) - n_old_classes - 1].permute(1, 0, 2, 3)[:,
                        small_label == cls].permute(1, 0),
                        prototypes[int(cls) - n_old_classes - 1]
                    )
                    angle = torch.acos(cos_similarity.clamp(-1, 1))

                    angle_noise[int(cls) - n_old_classes - 1] = \
                        angle_noise[int(cls) - n_old_classes - 1] + (angle ** 2).sum()
                    cnt[int(cls) - n_old_classes - 1] = \
                        cnt[int(cls) - n_old_classes - 1] + (small_label == cls).sum()

                self.progress(self.logger, batch_idx, len(self.train_loader))

            angle_noise = torch.sqrt(angle_noise / cnt)

            if config['data_loader']['args']['task']['step'] == 0:
                self.angle_noise = angle_noise
            else:
                self.angle_noise = torch.cat([self.prev_angle_noise, angle_noise], dim=0)

    def test(self):
        result = self._test()
        
        if self.rank == 0:
            log = {}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

    def progress(self, logger, i, total_length):
        period = total_length // 5
        if period == 0:
            return
        elif (i % period == 0):
            logger.info(f'[{i}/{total_length}]')

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.model).__name__
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
            }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _save_best_model(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.model).__name__
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
                # 'config': self.config
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
                # 'config': self.config
            }
        best_path = str(self.checkpoint_dir / 'model_best.pth')
        torch.save(state, best_path)
        self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path, test=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        if not self.reset_best_mnt:
            self.mnt_best = checkpoint['monitor_best']

        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['state_dict'])
            # self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        if test is False:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

def label_to_one_hot(label, logit, n_old_classes, ignore_index=255):
    target = torch.zeros_like(logit, device='cuda').float()
    for cls_idx in label.unique():
        if cls_idx in [0, ignore_index]:
            continue
        target[:, int(cls_idx) - (n_old_classes + 1)] = (label == int(cls_idx)).float()
    return target