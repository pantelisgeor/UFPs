from src.config import set_layer_config
from src.helpers import load_checkpoint

from src._gen_efficientnet import *

def create_model(
        model_name='mnasnet_100',
        pretrained=None,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        positional_embedings='',
        **kwargs): # I can pass args here and they'll go evnetualy into GenEfficientNet

    model_kwargs = dict(num_classes=num_classes, 
                        in_chans=in_chans, 
                        pretrained=pretrained,
                        positional_embedings=positional_embedings,
                        **kwargs)

    if model_name in globals():
        create_fn = globals()[model_name]
        model = create_fn(**model_kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path and not pretrained:
        load_checkpoint(model, checkpoint_path)

    return model
