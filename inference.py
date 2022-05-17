import os
from typing import Any, Mapping, Text
import torch
from helper import parser
from network import models
from network import wrapper as wrapper_lib
from utils.setup_helper import *
from data.utae_dynamicen import DynamicEarthNet


def main(opts: Any, config: Mapping[Text, Any]) -> None:
    """Runs the inference.

    Args:
        opts (Any): General options specifying the inference configuration.
        config (Mapping[Text, Any]): The detailed configuration.
    """    
    log = open(log_file, 'a')
    log_print = lambda ms: parse.log(ms, log)

    #networks define
    network = models.define_model(config, model=config['NETWORK']['NAME'], gpu_ids=opts.gpu_ids)
    print (network)
    print ('NUMBER OF PARAMS', sum(p.numel() for p in network.parameters() if p.requires_grad))
    log_print('Init {} as network'.format(config['NETWORK']['NAME']))

    # Initialize network wrapper
    if opts.resume:
        opts.checkpoint = os.path.join('/storage/www/user/toker/dynnet_ckpt/3dconv/weekly', 'best_ckpt.pth')


    # Configure data loader
    data_config = config['DATA']
    val_data = DynamicEarthNet(root=data_config['ROOT'], mode=data_config['EVAL_SUBSET'], type=data_config['TYPE'],
                               crop_size=data_config['RESIZE'], num_classes=data_config['NUM_CLASSES'],
                               ignore_index=data_config['IGNORE_INDEX'])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['EVALUATION']['BATCH_SIZE'], shuffle=False,
                                             num_workers=data_config['NUM_WORKERS'], pin_memory=True)

    iter_per_epoch = len(val_loader)
    wrapper = wrapper_lib.NetworkWrapper(network, iter_per_epoch, opts, config)
    log_print(
        'Load datasets from {}: val_set={}'.format(data_config['ROOT'], len(val_data)))

    metrics = wrapper.eval_model(1, val_loader, log_print)
    log_print('meanIoU:{:.10f}, pixelacc:{:.10f}, c:{}'.format(metrics.mIoU, metrics.pixAcc, metrics.c))


if __name__ == '__main__':
    parse = parser.Parser()
    opt, log_file = parse.parse()
    opt.is_Train = False
    make_deterministic(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpu_ids)
    config = parser.read_yaml_config(opt.config)

    main(opts=opt, config=config)