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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.F.adaptive_avg_pool1d,
    r"""
adaptive_avg_pool1d(input, output_size) -> Tensor

Applies a 1D adaptive average pooling over an input signal composed of
several input planes.

See :class:`~oneflow.nn.AdaptiveAvgPool1d` for details and output shape.

Args:
    output_size: the target output size (single integer)
""",
)


add_docstr(
    oneflow.F.adaptive_avg_pool2d,
    r"""
adaptive_avg_pool2d(input, output_size) -> Tensor

Applies a 2D adaptive average pooling over an input signal composed of
    several input planes.

See :class:`~oneflow.nn.AdaptiveAvgPool2d` for details and output shape.

Args:
    output_size: the target output size (single integer or
        double-integer tuple)
""",
)

add_docstr(
    oneflow.F.adaptive_avg_pool3d,
    r"""
adaptive_avg_pool3d(input, output_size) -> Tensor

Applies a 3D adaptive average pooling over an input signal composed of
    several input planes.

See :class:`~oneflow.nn.AdaptiveAvgPool3d` for details and output shape.

Args:
    output_size: the target output size (single integer or
        triple-integer tuple)
""",
)


