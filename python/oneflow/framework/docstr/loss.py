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
    oneflow.F.smooth_l1_loss,  
    r"""
    smooth_l1_loss(logits: Tensor, label, *, beta) -> Tensor 


    Function that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.
    
    See
    :class:`~oneflow.nn.SMOOTHL1LOSS` for more details.
 
    """
)
