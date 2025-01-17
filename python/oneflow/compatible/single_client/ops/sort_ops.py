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
from typing import Optional

import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework import id_util as id_util
from oneflow.compatible.single_client.framework import remote_blob as remote_blob_util
from oneflow.compatible.single_client.ops.transpose_util import (
    get_inversed_perm,
    get_perm_when_transpose_axis_to_last_dim,
)


def _sort_at_last_dim(
    input: oneflow._oneflow_internal.BlobDesc,
    direction: str = "ASCENDING",
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    assert direction in ["ASCENDING", "DESCENDING"]
    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Sort_"))
        .Op("sort")
        .Input("in", [input])
        .Output("out")
        .Attr("direction", direction)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def sort(
    input: oneflow._oneflow_internal.BlobDesc,
    axis: int = -1,
    direction: str = "ASCENDING",
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator sorts the input Blob at specified axis.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): A Blob
        axis (int, optional): dimension to be sorted. Defaults to the last dim (-1)
        direction (str, optional): The direction in which to sort the Blob values. If the direction is "ASCENDING", The order of input will be sorted as ascending, else, the order of input will be sorted as descending. Defaults to "ASCENDING".
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The sorted Blob

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import numpy as np
        import oneflow.compatible.single_client.typing as tp


        @flow.global_function()
        def sort_Job(x: tp.Numpy.Placeholder((5, ))
        ) -> tp.Numpy:
            return flow.sort(input=x)

        x = np.array([10, 2, 9, 3, 7]).astype("float32")
        out = sort_Job(x)

        # out [ 2.  3.  7.  9. 10.]

    """
    assert direction in ["ASCENDING", "DESCENDING"]
    name = name if name is not None else id_util.UniqueStr("Sort_")
    num_axes = len(input.shape)
    axis = axis if axis >= 0 else axis + num_axes
    assert 0 <= axis < num_axes, "axis out of range"
    if axis == num_axes - 1:
        return _sort_at_last_dim(input, direction, name)
    else:
        perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
        x = flow.transpose(input, perm, False, True, name + "_transpose")
        x = _sort_at_last_dim(x, direction, name)
        return flow.transpose(
            x, get_inversed_perm(perm), False, True, name + "_inverse_transpose"
        )


def _argsort_at_last_dim(
    input: oneflow._oneflow_internal.BlobDesc,
    direction: str = "ASCENDING",
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    assert direction in ["ASCENDING", "DESCENDING"]
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("ArgSort_")
        )
        .Op("arg_sort")
        .Input("in", [input])
        .Output("out")
        .Attr("direction", direction)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def argsort(
    input: oneflow._oneflow_internal.BlobDesc,
    axis: int = -1,
    direction: str = "ASCENDING",
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator sorts the input Blob at specified axis and return the indices of the sorted Blob.

    Args:
        input (oneflow._oneflow_internal.BlobDesc): A Blob
        axis (int, optional): dimension to be sorted. Defaults to the last dim (-1)
        direction (str, optional): The direction in which to sort the Blob values. If the direction is "ASCENDING", The order of input will be sorted as ascending, else, the order of input will be sorted as descending. Defaults to "ASCENDING".
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The indices of the sorted Blob

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import numpy as np
        import oneflow.compatible.single_client.typing as tp


        @flow.global_function()
        def argsort_Job(x: tp.Numpy.Placeholder((5, ))
        ) -> tp.Numpy:
            return flow.argsort(input=x)

        x = np.array([10, 2, 9, 3, 7]).astype("float32")
        out = argsort_Job(x)

        # out [1 3 4 2 0]

    """
    assert direction in ["ASCENDING", "DESCENDING"]
    name = name if name is not None else id_util.UniqueStr("ArgSort_")
    num_axes = len(input.shape)
    axis = axis if axis >= 0 else axis + num_axes
    assert 0 <= axis < num_axes, "axis out of range"
    if axis == num_axes - 1:
        return _argsort_at_last_dim(input, direction, name)
    else:
        perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
        x = flow.transpose(input, perm, False, True, name + "_transpose")
        x = _argsort_at_last_dim(x, direction, name)
        return flow.transpose(
            x, get_inversed_perm(perm), False, True, name + "_inverse_transpose"
        )
