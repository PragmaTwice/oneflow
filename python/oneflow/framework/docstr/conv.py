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
    oneflow.F.conv1d,
    r"""
    conv1d(x: Tensor) -> Tensor

    demo only

    .. math::

        \text{y}_{i} = \sin(\text{x}_{i})

    """,
)
add_docstr(
    oneflow.F.conv2d,
    r"""
    conv2d(x: Tensor) -> Tensor

    just demo
    
    """,
)
