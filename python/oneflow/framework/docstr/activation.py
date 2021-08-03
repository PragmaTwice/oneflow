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
    oneflow.F.prelu,  
    r"""
    prelu(x: Tensor, alpha: Tensor) -> Tensor 

   Applies the element-wise function:

    .. math::
        prelu(x) = \\max(0,x) + a * \\min(0,x)

    See
    :class:`~oneflow.nn.PReLU` for more details.
 
    """
)

add_docstr(
    oneflow.F.relu,
    r"""
    relu(x: Tensor, inplace: bool =False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~oneflow.nn.ReLU` for more details.

    """,
)
add_docstr(
    oneflow.F.hardsigmoid,
    r"""
hardsigmoid(x: Tensor)-> Tensor

Applies the element-wise function

.. math::
    \text{Hardsigmoid}(x) = \begin{cases}
        0 & \text{if~} x \le -3, \\
        1 & \text{if~} x \ge +3, \\
        x / 6 + 1 / 2 & \text{otherwise}
    \end{cases}

Args:
    inplace: If set to ``True``, will do this operation in-place. Default: ``False``

See :class:`~oneflow.nn.Hardsigmoid` for more details.
    """,
)
add_docstr(
    oneflow.F.hardswish,
    r"""
hardswish(x: Tensor)-> Tensor

Applies the hardswish function, element-wise, as described in the paper:

`Searching for MobileNetV3`_.

.. math::
    \text{Hardswish}(x) = \begin{cases}
        0 & \text{if~} x \le -3, \\
        x & \text{if~} x \ge +3, \\
        x \cdot (x + 3) /6 & \text{otherwise}
    \end{cases}

See :class:`~oneflow.nn.Hardswish` for more details.

.. _`Searching for MobileNetV3`:
    https://arxiv.org/abs/1905.02244
    """,
)


