import os
import time
from typing import Any, Text, Mapping
import torch, torchvision
from data.utae_dynamicen import DynamicEarthNet
from helper import parser
from network import models, pad
from network import wrapper as wrapper_lib
from utils.setup_helper import *

def main(opts: Any, config: Mapping[Text, Any]) -> None:
    """Runs the training.
    Args:
        opts (Any): Options specifying the training configuration.
    """
    log = open(log_file, 'a')
    log_print = lambda ms: parse.log(ms, log)

    # Define network
    network = models.define_model(config, model=config['NETWORK']['NAME'], gpu_ids=opts.gpu_ids)
    log_print('Init {} as network'.format(config['NETWORK']['NAME']))

    # Configure data loader
    data_config = config['DATA']
    train_data = DynamicEarthNet(root=data_config['ROOT'], mode=data_config['TRAIN_SUBSET'], type=data_config['TYPE'],
                               crop_size=data_config['RESIZE'], num_classes=data_config['NUM_CLASSES'],
                               ignore_index=data_config['IGNORE_INDEX'])
    val_data = DynamicEarthNet(root=data_config['ROOT'], mode=data_config['EVAL_SUBSET'], type=data_config['TYPE'],
                             crop_size=data_config['RESIZE'], num_classes=data_config['NUM_CLASSES'],
                             ignore_index=data_config['IGNORE_INDEX'])

    collate_fn = lambda x: pad.pad_collate(x, pad_value=config['NETWORK']['PAD_VALUE'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['TRAINING']['BATCH_SIZE'],
                                               shuffle=True, num_workers=data_config['NUM_WORKERS'],
                                               pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['EVALUATION']['BATCH_SIZE'],
                                             shuffle=False, num_workers=data_config['NUM_WORKERS'],
                                             pin_memory=True, collate_fn=collate_fn)

    iter_per_epoch = len(train_loader)
    wrapper = wrapper_lib.NetworkWrapper(network, iter_per_epoch, opts, config)

    log_print(
        'Load datasets from {}: train_set={} val_set={}'.format(data_config['ROOT'], len(train_data), len(val_data)))

    best_acc = wrapper.best_acc
    n_epochs = config['TRAINING']['EPOCHS']
    log_print('Start training from epoch {} to {}, best acc: {}'.format(opts.start_epoch, n_epochs, best_acc))
    for epoch in range(opts.start_epoch, n_epochs):
        start_time = time.time()
        log_print('>>> Epoch {}'.format(epoch))
        wrapper.train_epoch(epoch, train_loader, log_print)
        wrapper.save_ckpt(epoch, os.path.dirname(log_file), best_acc=best_acc, last_ckpt=True)

        # Save network periodically
        if (epoch + 1) % opts.save_step == 0:
            wrapper.save_ckpt(epoch, os.path.dirname(log_file), best_acc=best_acc)

        # Eval on validation set
        metrics = wrapper.eval_model(epoch, val_loader, log_print)
        log_print('\nEvaluate {} \npixAcc:{:.2f} meanIoU:{:.2f}'.format(epoch + 1, metrics.pixAcc, metrics.mIoU))

        # Save the best model
        mean_iou = metrics.mIoU
        if mean_iou > best_acc:
            best_acc = mean_iou
            wrapper.save_ckpt(epoch, os.path.dirname(log_file), best_acc=best_acc, is_best=True)
            log_print('>>Save best model: epoch={} best_iou:{:.2f}'.format(epoch + 1, mean_iou))

        #Program statistics
        rss, vms = get_sys_mem()
        max_gpu_mem = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        log_print('Memory usage: rss={:.2f}GB vms={:.2f}GB MaxGPUMem:{:.2f}GB Time:{:.2f}s'.format(rss, vms, max_gpu_mem, time.time() - start_time))

if __name__ == '__main__':
    parse = parser.Parser()
    opt, log_file = parse.parse()
    opt.is_Train = True
    make_deterministic(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpu_ids)

    config = parser.read_yaml_config(opt.config)
    main(opts=opt, config=config)