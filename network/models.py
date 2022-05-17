import torch
import warnings
from network.utae_seg import utae, unet3d, unet
warnings.filterwarnings("ignore")
from network.weight_init import weight_init

def init_net(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        print("Let's use", len(gpu_ids), "GPUs!")
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    net.apply(weight_init)
    return net

# Defines the network
def define_model(config, model, gpu_ids=[]):

    if model == 'unet':
        net = unet.Single_UNET(input_dim=4,
                encoder_widths=config['NETWORK']['ENCODER_WIDTHS'],
                decoder_widths=config['NETWORK']['DECODER_WIDTHS'],
                str_conv_k=config['NETWORK']['STR_CONV_K'],
                str_conv_s=config['NETWORK']['STR_CONV_S'],
                str_conv_p=config['NETWORK']['STR_CONV_P'],
                encoder_norm=config['NETWORK']['ENCODER_NORM'],
                pad_value=None,
                padding_mode=config['NETWORK']['PADDING_MODE'],)
    elif model=='utae':
        net = utae.UTAE(input_dim=4,
                encoder_widths=config['NETWORK']['ENCODER_WIDTHS'],
                decoder_widths=config['NETWORK']['DECODER_WIDTHS'],
                str_conv_k=config['NETWORK']['STR_CONV_K'],
                str_conv_s=config['NETWORK']['STR_CONV_S'],
                str_conv_p=config['NETWORK']['STR_CONV_P'],
                agg_mode=config['NETWORK']['AGG_MODE'],
                encoder_norm=config['NETWORK']['ENCODER_NORM'],
                n_head=config['NETWORK']['N_HEAD'],
                d_model=config['NETWORK']['D_MODEL'],
                d_k=config['NETWORK']['D_K'],
                encoder=False,
                return_maps=False,
                pad_value=config['NETWORK']['PAD_VALUE'],
                padding_mode=config['NETWORK']['PADDING_MODE'],)
    elif model =='uconvlstm':
        net = utae.RecUNet(
            input_dim=4,
            encoder_widths=config['NETWORK']['ENCODER_WIDTHS'],
            decoder_widths=config['NETWORK']['DECODER_WIDTHS'],
            str_conv_k=config['NETWORK']['STR_CONV_K'],
            str_conv_s=config['NETWORK']['STR_CONV_S'],
            str_conv_p=config['NETWORK']['STR_CONV_P'],
            temporal="lstm",
            encoder_norm=config['NETWORK']['ENCODER_NORM'],
            hidden_dim=64,
            encoder=False,
            padding_mode="zeros",
            pad_value=config['NETWORK']['PAD_VALUE'],
        )
    elif model == 'unet3d':
        net = unet3d.UNet3D(
                in_channel=4, n_classes=config['DATA']['NUM_CLASSES'], pad_value=config['NETWORK']['PAD_VALUE']
            )
    else:
        raise NotImplementedError('The model name [%s] is not recognized' % model)
    return init_net(net, gpu_ids=gpu_ids)


