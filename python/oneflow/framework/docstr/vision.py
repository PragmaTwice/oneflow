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
    oneflow.F.pad,
    r"""
    pad(input: Tensor, pad: List[int], mode: str = "constant", value: Scalar = 0) -> Tensor
    Pads tensor.

    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(\text{padding\_left}, \text{padding\_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom}`
        :math:`\text{padding\_front}, \text{padding\_back})`.

    Padding mode:
        See :class:`oneflow.nn.ConstantPad2d`, :class:`ooneflow.nn.ReflectionPad2d`, and
        :class:`ooneflow.nn.ReplicationPad2d` for concrete examples on how each of the
        padding modes works. Constant padding is implemented for arbitrary dimensions.
        Replicate and reflection padding is implemented for padding the last 3
        dimensions of 5D input tensor, or the last 2 dimensions of 4D input
        tensor, or the last dimension of 3D input tensor.


    Args:
        input (Tensor): N-dimensional tensor
        pad (List[int]): m-elements List, where
            :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

    """,
)
add_docstr(
    oneflow.F.upsample,
    r"""
    upsample(x: Tensor, height_scale: Float, width_scale: Float, align_corners: Bool,interpolation: str,data_format: str = "channels_first") -> Tensor
  
    Upsample a given multi-channel 2D (spatial) data.

    The input data is assumed to be of the form
    `minibatch x channels x height x width`.
    Hence, for spatial inputs, we expect a 4D Tensor.

    The algorithms available for upsampling are nearest neighbor,
    bilinear, 4D input Tensor, respectively.

    Args:
        height_scale (float):
            multiplier for spatial size. Has to match input size if it is a tuple.

        align_corners (bool): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is ``'bilinear'``.            

        interpolation (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'bilinear'``.        

        data_format (str, optional): Default: ``'channels_first'``

    Shape:
        - Input: : :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` , where
  
    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{height_scale} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{width_scale} \right\rfloor

  
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> input = flow.Tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)  
        >>> output = flow.F.upsample(input, height_scale=2.0, width_scale=2.0, align_corners=False, interpolation="nearest")
    
        >>> output #doctest: +ELLIPSIS
        tensor([[[[1., 1., 2., 2.],
                  [1., 1., 2., 2.],
                  [3., 3., 4., 4.],
                  [3., 3., 4., 4.]]]], dtype=oneflow.float32)

    See :class:`~oneflow.nn.Upsample` for more details.

    """,
)