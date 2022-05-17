import os
from argparse import Namespace
import numpy as np
import torch
from utils.metrics import eval_metrics_old
from utils.setup_helper import load_weights
from network import losses
from data.utae_dynamicen import *
from utils.palette import color_map
from matplotlib import pyplot

class NetworkWrapper:
    def __init__(self, net, iter_per_epoch, opt, config):
        self.net = net
        self.iter_per_epoch = iter_per_epoch
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        optim_dict, lr_dict, start_epoch = self.load_ckpt(opt)
        opt.start_epoch = start_epoch
        self.set_optimizer(opt, config, optim_dict, lr_dict)
        self.best_acc = 0.0
        self.config = config
        self.semantic_seg_loss = losses.create_loss(self.config['LOSS'])
        self.num_classes = self.config['NETWORK']['OUTPUT_CHANNELS']

    def set_optimizer(self, opt, config, optim_dict, lr_dict):
        if not opt.is_Train:
            return

        # Initialize optimizer
        lr = config['TRAINING']['LR']
        weight_decay = config['TRAINING']['WEIGHT_DECAY']
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        print('Setup Adam optimizer(lr={},wd={})'.format(lr, weight_decay))

        # Reload optimizer state dict if exists
        if optim_dict:
            self.optimizer.load_state_dict(optim_dict)

        # Initialize lrd scheduler
        self.lr_scheduler = None

    def recursive_todevice(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, dict):
            return {k: self.recursive_todevice(v) for k, v in x.items()}
        else:
            return [self.recursive_todevice(c) for c in x]

    def optim_step_(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_epoch(self, epoch, data_loader, log_print):
        self.net.train()
        epoch_loss = []
        for i, data in enumerate(data_loader):
            (image, dates), mask = self.recursive_todevice(data)
            mask = mask.long()
            pred = self.net(image, batch_positions=dates)
            loss = self.semantic_seg_loss(pred, mask)
            self.optim_step_(loss)
            epoch_loss.append(loss.item())
            if (i+1) % 50 == 0 or (i+1) == len(data_loader):
                log_print('Batch:{} TRAIN loss={:.3f}'.format(i + 1, np.mean(epoch_loss)))
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def eval_model(self, epoch, data_loader, log_print):
        """Evaluate the model
                METRICS: pixelAcc, meanIoU, classIoU
        """
        self.net.eval()
        epoch_val_loss = []
        total_correct, total_label = 0, 0
        total_inter, total_union = 0, 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                (image,dates), mask = self.recursive_todevice(data)
                print ('IMAGE SHAPE', image.shape)
                mask = mask.long()
                pred = self.net.forward(image, batch_positions=dates)
                val_loss = self.semantic_seg_loss(pred, mask)
                epoch_val_loss.append(val_loss.item())
                correct, labeled, inter, union = eval_metrics_old(pred, mask, self.num_classes, self.config['DATA']['IGNORE_INDEX'])
                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                # PRINT INFO
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()

                seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                           "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))}

                if (i+1) % 10 == 0 or (i+1) == len(data_loader):
                    log_print('Epoch:{} Batch:{} VAL loss={:.3f}, pixAcc={:.3f}, mIoU={:.3f}'.format(epoch,
                        i + 1, np.mean(epoch_val_loss), seg_metrics["Pixel_Accuracy"], seg_metrics["Mean_IoU"]))

            metrics = Namespace(pixAcc=pixAcc, mIoU=mIoU, c=seg_metrics["Class_IoU"])
        return metrics

    def save_ckpt(self, epoch, out_dir, last_ckpt=False, best_acc=None, is_best=False):
        ckpt = {'last_epoch': epoch, 'best_acc': best_acc, 'model_dict': self.net.state_dict(),
                'optimizer_dict': self.optimizer.state_dict(),
                'lr_scheduler_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None}

        if last_ckpt:
            ckpt_name = 'last_ckpt.pth'
        elif is_best:
            ckpt_name = 'best_ckpt.pth'
        else:
            ckpt_name = 'ckpt_ep{}.pth'.format(epoch + 1)
        ckpt_path = os.path.join(out_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

    def load_ckpt(self, config):
        ckpt_path = config.checkpoint
        if ckpt_path is None:
            return None, None, 0

        ckpt = load_weights(ckpt_path, self.device)
        start_epoch = ckpt['last_epoch'] + 1
        self.best_acc = ckpt['best_acc']
        print(
            'Load ckpt from {}, reset start epoch {}, best acc {}'.format(ckpt_path, config.start_epoch, self.best_acc))

        # Load net state
        model_dict = ckpt['model_dict']
        if len(model_dict.items()) == len(self.net.state_dict()):
            print('Reload all net parameters from weights dict')
            self.net.load_state_dict(model_dict)
        else:
            print('Reload part of net parameters from weights dict')
            self.net.load_state_dict(model_dict, strict=False)


        # Load optimizer state
        return ckpt['optimizer_dict'], ckpt['lr_scheduler_dict'], start_epoch

    def print_net_(self):
        for k, v in self.net.state_dict().items():
            print(k, v.size())
