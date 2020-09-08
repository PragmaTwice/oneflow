"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import string
import random

import numpy as np

from oneflow.python.onnx.load.handler import BackendHandler
from oneflow.python.onnx.load.handler import flow_func
from oneflow.python.onnx.load.handler import onnx_op
from oneflow.python.onnx.load.handlers.conv_mixin import ConvMixin
from oneflow.python.ops import nn_ops
from oneflow.python.ops import math_ops
from oneflow.python.ops import layers


@onnx_op("Conv")
class Conv(ConvMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.conv(node, tensor_dict)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls.conv(node, tensor_dict)


@onnx_op("BatchNormalization")
@flow_func(layers.batch_normalization)
class BatchNormalization(BackendHandler):
    @classmethod
    def get_attrs_processor_param(cls):
        return {
            "default": {"epsilon": 1e-5},
        }

    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        def randomString(stringLength=8):
            letters = string.ascii_lowercase
            return "".join(random.choice(letters) for i in range(stringLength))

        name = "bn_" + randomString()

        # update oneflow layers.batch_normalization to avoid this
        # it does not work on model with mulitple bn
        cls.copy_variable_file(node.input_tensor_names[1], name + "-gamma")
        cls.copy_variable_file(node.input_tensor_names[2], name + "-beta")
        cls.copy_variable_file(node.input_tensor_names[3], name + "-moving_mean")
        cls.copy_variable_file(node.input_tensor_names[4], name + "-moving_variance")
        node.input_tensor_names = node.input_tensor_names[:1]

        return [
            cls.run_onnx_node(node, tensor_dict, name=name, **kwargs, attrs={"axis": 1})
        ]

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


class PoolMixin(object):
    @classmethod
    def pool(cls, node, input_dict, pooling_type, strict=True):
        x = input_dict[node.input_tensor_names[0]]
        orig_x = x

        kernel_shape = node.attrs["kernel_shape"]

        spatial_size = len(kernel_shape)
        x_rank = spatial_size + 2

        kernel_shape = node.attrs["kernel_shape"]
        strides = node.attrs.get("strides", [1] * spatial_size)
        dilations = node.attrs.get("dilations", [1] * spatial_size)
        ceil_mode = bool(node.attrs.get("ceil_mode", 0))
        if ceil_mode != 0:
            raise ValueError("ceil_mode != 0 is not supported")
        pads = node.attrs.get("auto_pad", "NOTSET")
        if pads == "NOTSET":
            pads = node.attrs.get("pads", [0] * spatial_size * 2)
            pads = np.reshape(pads, [2, spatial_size]).T.tolist()
            pads = [[0, 0], [0, 0]] + pads

        count_include_pad = bool(node.attrs.get("count_include_pad", 0))
        if count_include_pad != 0:
            raise ValueError("count_include_pad != 0 is not supported")
        if pooling_type == "AVG":
            op = nn_ops.avg_pool2d
        elif pooling_type == "MAX":
            op = nn_ops.max_pool2d
        elif pooling_type == "MAX_WITH_ARGMAX":
            raise ValueError("maxpooling with argmax is not supported")

        if spatial_size != 2:
            raise ValueError("non-2d pooling is not supported")
        if node.attrs.get("storage_order", 0) != 0:
            raise ValueError("storage_order != 0 is not supported")

        return op(
            x, ksize=kernel_shape, strides=strides, padding=pads, data_format="NCHW"
        )


@onnx_op("AveragePool")
class AveragePool(PoolMixin, BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.pool(node, tensor_dict, "AVG", kwargs.get("strict", True))

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_10(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("MaxPool")
class MaxPool(PoolMixin, BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        pool_type = "MAX" if len(node.output_tensor_names) == 1 else "MAX_WITH_ARGMAX"
        return cls.pool(node, tensor_dict, pool_type, kwargs.get("strict", True))

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_8(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_10(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Relu")
@flow_func(math_ops.relu)
class Relu(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("SoftmaxCrossEntropyLoss")
@flow_func(nn_ops.sparse_softmax_cross_entropy_with_logits)
class SoftmaxCrossEntropyLoss(BackendHandler):
    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        inputs = (
            tensor_dict[node.input_tensor_names[1]],
            tensor_dict[node.input_tensor_names[0]],
        )
        return cls.run_onnx_node(node, tensor_dict, inputs=inputs, **kwargs)
