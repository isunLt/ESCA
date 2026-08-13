import os
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.model_zoo as model_zoo
from model.modules.resnet_encoder import resnet_encoders
import model.modules.dino.vision_transformer as dino_vit
import model.dinov2_vision_transformer as dinov2_vit

try:
    import mmcv
    from mmcv.runner import load_checkpoint
    from mmseg.apis import init_segmentor, inference_segmentor
    from mmseg.models import build_segmentor
except ImportError:
    mmcv = None

from functools import partial


_MEAN_PIXEL_IMAGENET = [0.485, 0.456, 0.406]
_STD_PIXEL_IMAGENET = [0.229, 0.224, 0.225]


def adapt_weights(architecture):
    if architecture == "imagenet" or architecture is None:
        return

    weights_url = {
        "moco_v2": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
        "moco_v1": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar",
        "swav": "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
        "deepcluster_v2": "https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar",
        "dino": "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
    }

    if not os.path.exists(f"weights/{architecture}.pt"):
        r = requests.get(weights_url[architecture], allow_redirects=True)
        os.makedirs("weights", exist_ok=True)
        with open(f"weights/{architecture}.pt", 'wb') as f:
            f.write(r.content)

    weights = torch.load(f"weights/{architecture}.pt")

    if architecture == "obow":
        return weights["network"]

    if architecture == "pixpro":
        weights = {
            k.replace("module.encoder.", ""): v
            for k, v in weights["model"].items()
            if k.startswith("module.encoder.")
        }
        return weights

    if architecture in ("moco_v1", "moco_v2", "moco_coco"):
        weights = {
            k.replace("module.encoder_q.", ""): v
            for k, v in weights["state_dict"].items()
            if k.startswith("module.encoder_q.") and not k.startswith("module.encoder_q.fc")
        }
        return weights

    if architecture in ("swav", "deepcluster_v2"):
        weights = {
            k.replace("module.", ""): v
            for k, v in weights.items()
            if k.startswith("module.") and not k.startswith("module.pro")
        }
        return weights

    if architecture == "dino":
        return weights


class Preprocessing:
    """
    Use the ImageNet preprocessing.
    """

    def __init__(self):
        normalize = T.Normalize(mean=_MEAN_PIXEL_IMAGENET, std=_STD_PIXEL_IMAGENET)
        self.preprocessing_img = normalize

    def __call__(self, image):
        return self.preprocessing_img(image)


class DilationFeatureExtractor(nn.Module):
    """
    Dilated ResNet Feature Extractor
    """
    # all = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    #        'resnet152']

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    def __init__(self, config, preprocessing=None):
        super(DilationFeatureExtractor, self).__init__()
        assert (
            config["images_encoder"] == "resnet50"
        ), "DilationFeatureExtractor is only available for resnet50"
        Encoder = resnet_encoders["resnet50"]["encoder"]
        params = resnet_encoders["resnet50"]["params"]
        params.update(replace_stride_with_dilation=[True, True, True])
        self.encoder = Encoder(**params)

        if config["image_weights"] == "imagenet":
            self.encoder.load_state_dict(model_zoo.load_url(self.model_urls["resnet50"]))

        weights = adapt_weights(architecture=config["image_weights"])
        if weights is not None:
            self.encoder.load_state_dict(weights)

        for param in self.encoder.parameters():
            param.requires_grad = False

        # for (name, param) in list(self.encoder.named_parameters()):  # [:-2]
        #     if 'layer4.2' in name:
        #         print('=============warning train 2d encoder!==================')
        #         continue
        #     param.requires_grad = False

        in1 = 2048

        self.decoder = nn.Sequential(
            nn.Conv2d(in1, config["model_n_out"], 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
        )
        self.preprocessing = preprocessing
        self.normalize_feature = config["normalize_features"]

        self.decoupled_head = config['decoupled_head']
        if self.decoupled_head:
            self.decoder_tmp = nn.Sequential(
            nn.Conv2d(in1, config["model_n_out"], 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
        )

    # def forward(self, x):
    #     if self.preprocessing:
    #         x = self.preprocessing(x)
    #     x = self.decoder(self.encoder(x))
    #     if self.normalize_feature:
    #         x = F.normalize(x, p=2, dim=1)
    #     return x
    def forward(self, x):
        if self.preprocessing:
            x = self.preprocessing(x)
        x = self.encoder(x)
        if self.decoupled_head:
            x_tmp = self.decoder_tmp(x)
        x = self.decoder(x)
        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=1)
            if self.decoupled_head:
                x_tmp = F.normalize(x_tmp, p=2, dim=1)
        if self.decoupled_head:
            return x, x_tmp
        return x


class PPKTFeatureExtractor(nn.Module):
    """
    PPKT baseline
    """

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    def __init__(self, config, preprocessing=None):
        super(PPKTFeatureExtractor, self).__init__()
        Encoder = resnet_encoders[config["images_encoder"]]["encoder"]
        params = resnet_encoders[config["images_encoder"]]["params"]
        self.encoder = Encoder(**params)

        if config["image_weights"] == "imagenet":
            self.encoder.load_state_dict(model_zoo.load_url(self.model_urls[config["images_encoder"]]))

        if config["image_weights"] not in (None, "imagenet"):
            assert (
                config["images_encoder"] == "resnet50"
            ), "{} weights are only available for resnet50".format(
                config["images_weights"]
            )
            weights = adapt_weights(architecture=config["image_weights"])
            if weights is not None:
                self.encoder.load_state_dict(weights)

        for param in self.encoder.parameters():
            param.requires_grad = False

        if config["images_encoder"] == "resnet18":
            in1 = 512
        elif config["images_encoder"] == "resnet50":
            in1 = 2048

        self.decoder = nn.Sequential(
            nn.Conv2d(in1, config["model_n_out"], 1),
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=True),
        )
        self.preprocessing = preprocessing
        self.normalize_feature = config["normalize_features"]

    def forward(self, x):
        if self.preprocessing:
            x = self.preprocessing(x)
        x = self.decoder(self.encoder(x))
        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=1)
        return x


class DinoVitFeatureExtractor(nn.Module):
    """
    DINO Vision Transformer Feature Extractor.
    """
    def __init__(self, config, preprocessing=None):
        super(DinoVitFeatureExtractor, self).__init__()
        dino_models = {
            "vit_small_p16": ("vit_small", 16, 384),
            "vit_small_p8": ("vit_small", 8, 384),
            "vit_base_p16": ("vit_base", 16, 768),
            "vit_base_p8": ("vit_base", 8, 768),
        }
        assert (
            config["images_encoder"] in dino_models.keys()
        ), f"DilationFeatureExtractor is only available for {dino_models.keys()}"

        model_name, patch_size, embed_dim = dino_models[config["images_encoder"]]

        print("Use Vision Transformer pretrained with DINO as the image encoder")
        print(f"==> model_name: {model_name}")
        print(f"==> patch_size: {patch_size}")
        print(f"==> embed_dim: {embed_dim}")

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.encoder = dino_vit.__dict__[model_name](patch_size=patch_size, num_classes=0)
        dino_vit.load_pretrained_weights(self.encoder, "", None, model_name, patch_size)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, config["model_n_out"], 1),
            nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=True),
        )
        self.preprocessing = preprocessing
        self.normalize_feature = config["normalize_features"]

    def forward(self, x):
        if self.preprocessing:
            x = self.preprocessing(x)
        batch_size, _, height, width = x.size()
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        f_height = height // self.patch_size
        f_width = width // self.patch_size

        x = self.encoder(x, all=True)
        # the output of x should be [batch_size x (1 + f_height * f_width) x self.embed_dim]
        assert x.size(1) == (1 + f_height * f_width)
        # Remove the CLS token and reshape the the patch token features.
        x = x[:, 1:, :].contiguous().transpose(1, 2).contiguous().view(batch_size, self.embed_dim, f_height, f_width)

        x = self.decoder(x)
        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=1)
        return x

class DinoV2FeatureExtractor(nn.Module):
    """
    DINO Vision Transformer Feature Extractor.
    """
    def __init__(self, config, preprocessing=None):
        super(DinoV2FeatureExtractor, self).__init__()
        dino_models = {
            "dinov2_small_p14": ("dinov2_vits14", 14, 384),
            "dinov2_base_p14": ("dinov2_vitb14", 14, 768),
            "dinov2_large_p14": ("dinov2_vitl14", 14, 1024),
            # "dinov2_small_ade20k_linear": ("dinov2_vits14", 'ade20k', 'linear'),
            # "dinov2_base_ade20k_linear": ("dinov2_vitb14", 'ade20k', 'linear'),
            # "dinov2_large_ade20k_linear": ("dinov2_vitl14", 'ade20k', 'linear'),
        }
        assert (
            config["images_encoder"] in dino_models.keys()
        ), f"DilationFeatureExtractor is only available for {dino_models.keys()}"


        model_name, patch_size, embed_dim = dino_models[config["images_encoder"]]

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # self.which_feature = config["image_backbone"]["feat"]
        print("Image teacher:")
        print(f"==> model_name: {model_name}")
        print(f"==> patch_size: {patch_size}")
        print(f"==> embed_dim: {embed_dim}")
        # assert config["point_backbone"]["nb_class"] == embed_dim

        # Compute feature size
        height, width = config["crop_size"]
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        self.f_height = height // self.patch_size
        self.f_width = width // self.patch_size

        # Load ViT
        self.encoder = dinov2_vit.__dict__[model_name](
            patch_size=patch_size,
            pretrained=True,
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        # self.patch_size = cfg['model']['backbone']['patch_size']  # 16
        # self.in_index = cfg['model']['decode_head']['in_index']
        # embed_dim = cfg['model']['decode_head']['channels'] * 4

        self.decoder = nn.Sequential(
            # nn.BatchNorm2d(embed_dim),
            # nn.Conv2d(embed_dim, config["model_n_out"], 1),
            nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=True),
        )
        self.preprocessing = preprocessing
        self.normalize_feature = config["normalize_features"]

    def forward(self, x):
        if self.preprocessing:
            x = self.preprocessing(x)
        batch_size = x.shape[0]

        output = self.encoder.forward_get_last_n(x)
        # feat = output[self.which_feature]
        # feat = output['x']
        feat = output['x_pre_norm']
        x = torch.cat(feat, dim=2)

        # Remove the CLS token and reshape the patch token features.
        x = (
            x[:, 1:, :]
            .transpose(1, 2).contiguous()
            .view(batch_size, self.embed_dim, self.f_height, self.f_width)
        )

        # Go through decoder
        x = self.decoder(x)
        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=1)
        return x

class DinoV2M2FFeatureExtractor(nn.Module):
    """
    DINO Vision Transformer Feature Extractor.
    """
    def __init__(self, config, preprocessing=None):
        super(DinoV2M2FFeatureExtractor, self).__init__()
        dino_models = {
            # "dinov2_small_p16": ("vits14", 16, 384),
            # "dinov2_base_p8": ("vitb14", 8, 384),
            # "dinov2_large_p16": ("vitl14", 16, 768),
            "dinov2_m2f_small_ade20k_linear": ("dinov2_vits14", 'ade20k', 'linear'),
            "dinov2_m2f_base_ade20k_linear": ("dinov2_vitb14", 'ade20k', 'linear'),
            "dinov2_m2f_large_ade20k_linear": ("dinov2_vitl14", 'ade20k', 'linear'),
        }
        assert (
            config["images_encoder"] in dino_models.keys()
        ), f"DilationFeatureExtractor is only available for {dino_models.keys()}"


        model_name, head_dataset, head_type = dino_models[config["images_encoder"]]

        print("Use Vision Transformer pretrained with DINO as the image encoder")
        print(f"==> model_name: {model_name}")
        print(f"==> head_dataset: {head_dataset}")
        print(f"==> head_type: {head_type}")

        backbone_model = torch.hub.load(repo_or_dir="./model", source='local', model=model_name)

        HEAD_SCALE_COUNT = 3

        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        local_url = '/data/stf/codes/SLidR/model/dinov2_old/configs/eval'
        head_config_url = f"{model_name}_{head_dataset}_{head_type}_config.py"
        head_checkpoint_url = f"{DINOV2_BASE_URL}/{model_name}/{model_name}_{head_dataset}_{head_type}_head.pth"

        CONFIG_PATH = os.path.join(local_url, head_config_url)
        cfg = mmcv.Config.fromfile(CONFIG_PATH)
        if head_type == "ms":
            cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
            print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

        model = backbone_model
        model.forward = partial(
            backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
        )
        model.init_weights()

        # load_checkpoint(model, head_checkpoint_url, map_location="cpu")
        self.encoder = model

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.patch_size = cfg['model']['backbone']['patch_size']  # 16
        self.in_index = cfg['model']['decode_head']['in_index']
        embed_dim = cfg['model']['decode_head']['channels'] * len(self.in_index)

        self.decoder = nn.Sequential(
            # nn.BatchNorm2d(embed_dim),
            # nn.Conv2d(embed_dim, config["model_n_out"], 1),
            nn.Upsample(scale_factor=self.patch_size, mode="bilinear", align_corners=True),
        )
        self.preprocessing = preprocessing
        self.normalize_feature = config["normalize_features"]

    def forward(self, x):
        if self.preprocessing:
            x = self.preprocessing(x)
        batch_size, _, height, width = x.size()
        # assert (height % self.patch_size) == 0
        # assert (width % self.patch_size) == 0
        # f_height = height // self.patch_size
        # f_width = width // self.patch_size

        x = self.encoder(x)
        x_h, x_w = x[0].size(2), x[0].size(3)
        x = [F.interpolate(x_i, size=(x_h, x_w), mode='bilinear', align_corners=True) for x_i in x]
        x = torch.cat(x, dim=1)
        # # the output of x should be [batch_size x (1 + f_height * f_width) x self.embed_dim]
        # assert x.size(1) == (1 + f_height * f_width)
        # # Remove the CLS token and reshape the the patch token features.
        # x = x[:, 1:, :].contiguous().transpose(1, 2).contiguous().view(batch_size, self.embed_dim, f_height, f_width)

        x = self.decoder(x)
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=1)
        return x

class DilationSegmentor(nn.Module):
    """
    Dilated ResNet Feature Extractor
    """
    # all = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    #        'resnet152']

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    def __init__(self, config, preprocessing=None):
        super(DilationSegmentor, self).__init__()
        assert (
            config["images_encoder"] == "resnet50"
        ), "DilationFeatureExtractor is only available for resnet50"
        Encoder = resnet_encoders["resnet50"]["encoder"]
        params = resnet_encoders["resnet50"]["params"]
        params.update(replace_stride_with_dilation=[True, True, True])
        self.encoder = Encoder(**params)

        if config["image_weights"] == "imagenet":
            self.encoder.load_state_dict(model_zoo.load_url(self.model_urls["resnet50"]))

        weights = adapt_weights(architecture=config["image_weights"])
        if weights is not None:
            self.encoder.load_state_dict(weights)

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        in1 = 2048

        self.decoder = nn.Sequential(
            nn.Conv2d(in1, config["model_n_out"], 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
        )
        self.preprocessing = preprocessing
        self.normalize_feature = config["normalize_features"]

    def forward(self, x):
        if self.preprocessing:
            x = self.preprocessing(x)
        x = self.decoder(self.encoder(x))
        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=1)
        return x


class DinoV2BNHeadSegmentor(nn.Module):
    """
    DINO Vision Transformer Feature Extractor.
    """
    def __init__(self, config, preprocessing=None):
        super(DinoV2BNHeadSegmentor, self).__init__()
        dino_models = {
            # "dinov2_small_p16": ("vits14", 16, 384),
            # "dinov2_base_p8": ("vitb14", 8, 384),
            # "dinov2_large_p16": ("vitl14", 16, 768),
            "dinov2_large_ade20k_linear": ("dinov2_vitl14", 'ade20k', 'linear'),
        }
        assert (
            config["images_encoder"] in dino_models.keys()
        ), f"DilationFeatureExtractor is only available for {dino_models.keys()}"

        model_name, head_dataset, head_type = dino_models[config["images_encoder"]]

        print("Use Vision Transformer pretrained with DINO as the image encoder")
        print(f"==> model_name: {model_name}")
        print(f"==> head_dataset: {head_dataset}")
        print(f"==> head_type: {head_type}")

        backbone_model = torch.hub.load(repo_or_dir="./model", source='local', model=model_name)
        # backbone_model.eval()
        # backbone_model.cuda()
        HEAD_SCALE_COUNT = 3

        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        local_url = '/data/stf/codes/SLidR/model/dinov2_old/configs/eval'
        head_config_url = f"{model_name}_{head_dataset}_{head_type}_config.py"
        head_checkpoint_url = f"{DINOV2_BASE_URL}/{model_name}/{model_name}_{head_dataset}_{head_type}_head.pth"

        CONFIG_PATH = os.path.join(local_url, head_config_url)
        cfg = mmcv.Config.fromfile(CONFIG_PATH)
        if head_type == "ms":
            cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
            print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

        # model = init_segmentor(cfg)
        model = build_segmentor(cfg.model, test_cfg=config.get('test_cfg'))
        model.backbone.forward = partial(
            backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
        )
        # model.init_weights()

        load_checkpoint(model, head_checkpoint_url, map_location="cpu")
        self.encoder = backbone_model
        self.encoder.forward = partial(
            backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
        )
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder_bnhead = model.decode_head
        for param in self.decoder_bnhead.parameters():
            param.requires_grad = False

        # self.patch_size = cfg['model']['backbone']['patch_size']  # 16
        self.in_index = cfg['model']['decode_head']['in_index']
        # embed_dim = cfg['model']['decode_head']['channels']
        self.superpixel_size = config['superpixel_size']

        self.decoder = nn.Sequential(
            nn.BatchNorm2d(self.superpixel_size),
            nn.Conv2d(self.superpixel_size, config["model_n_out"], 1),
            # nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
        )

        self.preprocessing = preprocessing
        self.normalize_feature = config["normalize_features"]

    def forward(self, x):
        if self.preprocessing:
            x = self.preprocessing(x)
        batch_size, _, height, width = x.size()
        # assert (height % self.patch_size) == 0
        # assert (width % self.patch_size) == 0
        # f_height = height // self.patch_size
        # f_width = width // self.patch_size

        x = self.encoder(x)
        # x_h, x_w = x[0].size(2), x[0].size(3)
        # x = [F.interpolate(x_i, size=(x_h, x_w), mode='bilinear', align_corners=True) for x_i in x]
        # x = [x[i] for i in self.in_index]
        # x = torch.cat(x, dim=1)
        # # the output of x should be [batch_size x (1 + f_height * f_width) x self.embed_dim]
        # assert x.size(1) == (1 + f_height * f_width)
        # # Remove the CLS token and reshape the the patch token features.
        # x = x[:, 1:, :].contiguous().transpose(1, 2).contiguous().view(batch_size, self.embed_dim, f_height, f_width)
        x = self.decoder_bnhead(x)
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
        # x = self.decoder(x)
        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=1)
        return x
