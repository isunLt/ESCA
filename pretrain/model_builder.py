import torch
from model import (
    MinkUNet,
    MinkUNet_S,
    MinkUNet18,
    MinkUNet50,
    MinkUNet101,
    VoxelNet,
    SPVCNN,
    DilationFeatureExtractor,
    PPKTFeatureExtractor,
    Preprocessing,
    DinoVitFeatureExtractor,
    DinoV2FeatureExtractor,
    Cylinder3D,
)


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    loaded_dict = {
        k.replace("module.", ""): v for k, v in loaded_dict.items()
    }
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        new_k = k
        if (
            new_k in loaded_dict and net_state_dict[k].size() == loaded_dict[new_k].size()
        ):
            new_loaded_dict[k] = loaded_dict[new_k]
        else:
            print("Skipped loading parameter {}".format(k))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net


def make_model(config, load_path=None):
    """
    Build points and image models according to what is in the config
    """
    if config["model_points"] == "voxelnet":
        model_points = VoxelNet(4, config["model_n_out"], config)
    elif config['model_points'] == 'spvcnn':
        model_points = SPVCNN(4, config["model_n_out"], config)
    elif config['model_points'] == 'minkunet_s':
        model_points = MinkUNet_S(1, config['model_n_out'], config)
    elif config['model_points'] == 'minkunet18':
        model_points = MinkUNet18(1, config['model_n_out'], config)
    elif config['model_points'] == 'minkunet50':
        model_points = MinkUNet50(1, config['model_n_out'], config)
    elif config['model_points'] == 'minkunet101':
        model_points = MinkUNet101(1, config['model_n_out'], config)
    elif config['model_points'] == 'cylinder3d':
        model_points = Cylinder3D(input_dim=4, output_dim=config["model_n_out"], cfg=config)
    else:
        model_points = MinkUNet(1, config["model_n_out"], config)

    if config["images_encoder"].find("vit_") != -1:
        model_images = DinoVitFeatureExtractor(config, preprocessing=Preprocessing())
    elif config["images_encoder"].find("dinov2_") != -1:
        model_images = DinoV2FeatureExtractor(config, preprocessing=Preprocessing())
    elif config["decoder"] == "dilation":
        model_images = DilationFeatureExtractor(config, preprocessing=Preprocessing())
    elif config["decoder"] == "ppkt":
        model_images = PPKTFeatureExtractor(config, preprocessing=Preprocessing())
    else:
        # model with a decoder
        raise Exception(f"Model not found: {config['decoder']}")
    if load_path:
        print("Training with pretrained model")
        checkpoint = torch.load(load_path, map_location="cpu")
        model_points = forgiving_state_restore(model_points, loaded_dict=checkpoint['model_points'])
        model_images = forgiving_state_restore(model_images, loaded_dict=checkpoint['model_images'])
    return model_points, model_images
