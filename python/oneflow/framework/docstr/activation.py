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
    Gelu activation operator.

    The equation is: Applies the element-wise function:

    .. math::
        prelu(x) = \\max(0,x) + alpha * \\min(0,x)
    For example:

     .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.PReLU()
        >>> input = flow.Tensor(np.asarray([[[[1, -2], [3, 4]]]]), dtype=flow.float32)
        >>> print(m(input).numpy())
        [[[[ 1.  -0.5]
           [ 3.   4. ]]]]

    See
    :class:`~oneflow.nn.PReLU` for more details.
 
    """
)
add_docstr(
    oneflow.F.gelu,  
    r"""
    gelu(x: Tensor) -> Tensor 

    The equation is:

    .. math::
        out = 0.5 * x * (1 + tanh(\\sqrt{\\frac{2}{\\pi}} * (x + 0.044715x^{3})))
    
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> gelu = flow.nn.GELU()

        >>> out = gelu(input)
        >>> out
        tensor([-0.1543,  0.    ,  0.3457], dtype=oneflow.float32)


    See    
    :class:`~oneflow.nn.GELU` for more details.
 
    """
)

add_docstr(
    oneflow.F.log_sigmoid,
    r"""
    log_sigmoid(x: Tensor) -> Tensor 

    Applies the element-wise function:

    .. math::
        \\text{log_sigmoid}(x) = \\log\\left(\\frac{ 1 }{ 1 + \\exp(-x)}\\right)

    For example:

    .. code-block:: python


        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> logsigmoid = flow.nn.LogSigmoid()

        >>> out = logsigmoid(input)
        >>> out
        tensor([-0.9741, -0.6931, -0.4741], dtype=oneflow.float32)

    See :class:`~oneflow.nn.LogSigmoid` for more details.

    """,
)

add_docstr(
    oneflow.F.softsign,
    r"""
    softsign(x: Tensor, *) -> Tensor 

    The formula is: 
    
    .. math::  
    
        softsign(x) = \frac{x}{1 + |x|}
 
    See :class:`~oneflow.nn.Softsign` for more details.
    
    """,
)


add_docstr(
    oneflow.F.tanh,
    r""" 
    tanh(x: Tensor) -> Tensor 

    This operator computes the hyperbolic tangent value of Tensor.

    The equation is:

    .. math::

        out = \\frac{e^x-e^{-x}}{e^x+e^{-x}}

    See :class:`~oneflow.nn.Tanh` for more details.
    
    """,
)

add_docstr(
    oneflow.F.silu,
    r""" 
    tanh(x: Tensor, *) -> Tensor 

    silu() activation:

    .. math::
    
        \text{silu}(x) = x * sigmoid(x)

    See :class:`~oneflow.nn.SiLU` for more details.
    
    """,
)



add_docstr(
    oneflow.F.mish,
    r""" 
    mish(x: Tensor, *) -> Tensor 

  Applies the element-wise function:

    .. math::
        \\text{Mish}(x) = x * \\text{Tanh}(\\text{Softplus}(x))


    See :class:`~oneflow.nn.Mish` for more details.
    
    """,
)


add_docstr(
    oneflow.F.layer_norm,
    r"""    
    layer_norm(x: Tensor, *, Int64 begin_norm_axis, Int64 begin_params_axis, Double epsilon) -> Tensor 

    Applies the element-wise function:

    .. math::
        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta


    See :class:`~oneflow.nn.LayerNorm` for more details.
    
    """,
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
add_docstr(
    oneflow.F.sigmoid,
    r"""
sigmoid(input) -> Tensor

Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

See :class:`~oneflow.nn.Sigmoid` for more details.

    """,
)



