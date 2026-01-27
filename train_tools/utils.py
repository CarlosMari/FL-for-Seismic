from .models import *
__all__ = ["create_models"]

MODELS = {
    "fedavg_cifar": fedavgnet.FedAvgNetCIFAR,
    "seismic": unet.UNet
}

NUM_CLASSES = {
    "cifar10": 10,
    "seismic": 6,
}


def create_models(model_name, dataset_name):
    """Create a network model"""

    num_classes = NUM_CLASSES[dataset_name]

    if dataset_name == 'seismic':
        ch = 1
    else:
        ch = 3

    model = MODELS[model_name](num_classes=num_classes, in_channels=ch)

    return model
