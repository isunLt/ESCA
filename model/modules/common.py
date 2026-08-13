# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections.abc as collections
from enum import Enum

import MinkowskiEngine as ME
import torch
import torch.nn as nn


class NormType(Enum):
    BATCH_NORM = 0
    SPARSE_LAYER_NORM = 1
    SPARSE_INSTANCE_NORM = 2
    SPARSE_SWITCH_NORM = 3
    # PDBatchNorm = 4


def get_norm(norm_type, n_channels, D, bn_momentum=0.1, conditions=None):
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.SPARSE_INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels, D=D)
    # elif norm_type == NormType.PDBatchNorm:
    #     return PDBatchNorm(n_channels, momentum=bn_momentum, conditions=conditions)
    else:
        raise ValueError(f"Norm type: {norm_type} not supported")


class PDBatchNorm(ME.MinkowskiBatchNorm):
    def __init__(
        self,
        num_features,
        eps=1e-3,
        momentum=0.01,
        conditions=("nuscenes", "semkitti", "waymo"),
        affine=True,
    ):
        super().__init__()
        assert conditions is not None
        self.conditions = conditions
        self.bn = nn.ModuleList(
            [
                nn.BatchNorm1d(
                    num_features=num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                )
                for _ in conditions
            ]
        )

    def forward(self, feat, condition):
        output = torch.zeros_like(feat.F, device=feat.F.device, dtype=feat.F.dtype)
        batch2index = [[] for _ in self.conditions]
        for cond, perm in zip(condition, feat.decomposition_permutations):
            batch2index[self.conditions.index(cond)].append(perm)
        for idx, b2i in enumerate(batch2index):
            b2i = torch.concat(b2i)
            output[b2i] = self.bn[idx](feat.F[b2i])
        if isinstance(feat, ME.TensorField):
            return ME.TensorField(
                output,
                coordinate_field_map_key=feat.coordinate_field_map_key,
                coordinate_manager=feat.coordinate_manager,
                quantization_mode=feat.quantization_mode,
            )
        else:
            return ME.SparseTensor(
                output,
                coordinate_map_key=feat.coordinate_map_key,
                coordinate_manager=feat.coordinate_manager,
            )


class ConvType(Enum):
    """
    Define the kernel region type
    """

    HYPERCUBE = 0, "HYPERCUBE"
    SPATIAL_HYPERCUBE = 1, "SPATIAL_HYPERCUBE"
    SPATIO_TEMPORAL_HYPERCUBE = 2, "SPATIO_TEMPORAL_HYPERCUBE"
    HYPERCROSS = 3, "HYPERCROSS"
    SPATIAL_HYPERCROSS = 4, "SPATIAL_HYPERCROSS"
    SPATIO_TEMPORAL_HYPERCROSS = 5, "SPATIO_TEMPORAL_HYPERCROSS"
    SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = 6, "SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS "

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


# Covert the ConvType var to a RegionType var
conv_to_region_type = {
    # kernel_size = [k, k, k, 1]
    ConvType.HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIO_TEMPORAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIO_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
}

int_to_region_type = {i: m[0] for i, m in enumerate(ME.RegionType.__entries.values())}


def convert_conv_type(conv_type, kernel_size, D):
    assert isinstance(conv_type, ConvType), "conv_type must be of ConvType"
    region_type = conv_to_region_type[conv_type]
    axis_types = None
    if conv_type == ConvType.SPATIAL_HYPERCUBE:
        # No temporal convolution
        if isinstance(kernel_size, collections.Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.HYPERCUBE:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIAL_HYPERCROSS:
        if isinstance(kernel_size, collections.Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.HYPERCROSS:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS:
        # Define the CUBIC conv kernel for spatial dims and CROSS conv for temp dim
        axis_types = [
            ME.RegionType.HYPER_CUBE,
        ] * 3
        if D == 4:
            axis_types.append(ME.RegionType.HYPER_CROSS)
    return region_type, axis_types, kernel_size


def conv(
    in_planes,
    out_planes,
    kernel_size,
    stride=1,
    dilation=1,
    bias=False,
    conv_type=ConvType.HYPERCUBE,
    D=-1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiConvolution(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def conv_tr(
    in_planes,
    out_planes,
    kernel_size,
    upsample_stride=1,
    dilation=1,
    bias=False,
    conv_type=ConvType.HYPERCUBE,
    D=-1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        upsample_stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiConvolutionTranspose(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=upsample_stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def sum_pool(kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPERCUBE, D=-1):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiSumPooling(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        kernel_generator=kernel_generator,
        dimension=D,
    )
