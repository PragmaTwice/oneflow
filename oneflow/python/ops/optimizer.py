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
from __future__ import absolute_import

import collections.abc

import oneflow as flow
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob
import oneflow.python.eager.gradient_util as gradient_util
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.core.job.job_conf_pb2 as job_conf_pb
from typing import Tuple, Optional, Union, Sequence, Text


class ClipGradientConf:
    @property
    def clip_conf(self) -> op_conf_pb.ClipConf:
        raise NotImplementedError()


@oneflow_export("optimizer.grad_clipping.by_global_norm")
class ClipByGlobalNorm(ClipGradientConf):
    r"""This operator limits the norm of `Input` with `clip_norm`. If the norm of `Input` less than the `clip_norm`, the `Output` will be the same as `Input`. If the norm of `Input` greater than the `clip_norm`, the `Output` will be scaled. 

    The equation is: 

    .. math:: 
    
        Out = \frac{clip\_norm*Input}{norm(Input)}
    
    Args:
        clip_norm (float): The maximum norm value. 

    For example:

    .. code-block:: python 
        
        import oneflow as flow
        import oneflow.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            # Set learning rate as 0.001
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
            # Set gradient_clip
            gradient_clip = flow.optimizer.grad_clipping.by_global_norm(1.0)
            # Set AdamW optimizer with gradient clip
            flow.optimizer.AdamW(lr_scheduler, 
                        do_bias_correction=False, weight_decay=0.00005, 
                        grad_clipping=gradient_clip).minimize(loss)

            return loss

    """
    def __init__(self, clip_norm):
        self.clip_norm = clip_norm

    @property
    def clip_conf(self):
        clip_conf = op_conf_pb.ClipConf()
        clip_conf.clip_by_global_norm.clip_norm = self.clip_norm
        return clip_conf


class WarmupConf:
    @property
    def warmup_conf(self) -> op_conf_pb.WarmupConf:
        raise NotImplementedError()


@oneflow_export("optimizer.warmup.constant")
class ConstantWarmup(WarmupConf):
    r"""This operator use the constant warmup strategy to adjust the learning rate. Before the steps specified by user, the learning rate is: 

    .. math::

        learning\_rate = base\_learning\_rate*multiplier

    After the steps specified by user, the learning rate is: 

    .. math:: 

        learning\_rate = base\_learning\_rate

    Args:
        steps (int): [description]
        multiplier (float): The scale factor :math:`multiplier`, it should be greater than 0. and less than 1.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        
        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            # Before 10 epochs, the learning rate is 0.001
            # After 10 epochs, the learning rate is 0.01
            warmup_scheduler = flow.optimizer.warmup.constant(10, 0.1)
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.01], warmup=warmup_scheduler)
            flow.optimizer.Adam(lr_scheduler).minimize(loss)

            return loss

    """
    def __init__(self, steps, multiplier):
        self.steps = steps
        self.multiplier = multiplier

    @property
    def warmup_conf(self) -> op_conf_pb.WarmupConf:
        warmup_conf = op_conf_pb.WarmupConf()
        warmup_conf.constant_conf.warmup_batches = self.steps
        warmup_conf.constant_conf.multiplier = self.multiplier
        return warmup_conf


@oneflow_export("optimizer.warmup.linear")
class LinearWarmup(WarmupConf):
    r"""This operator use the linear warmup strategy to adjust the learning rate.
    When current train step is less than warmup steps, the learning rate will be updated as: 

    .. math:: 

        & current\_multiplier = start\_multiplier + (1-start\_multiplier)*\frac{train\_step}{warmup\_step} 
        
        & current\_learning\_rate = learning\_rate*current\_multiplier

    Args:
        steps (int): The warmup steps. 
        start_multiplier (float): The start multiplier(:math:`start\_multiplier`). it should be greater than 0. and less than 1.
    
    For example: 

    .. code-block:: python  

        import oneflow as flow
        import oneflow.typing as tp
        
        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            # Before 10 epochs, the learning rate will increase from 0.001 to 0.01 in linear. 
            warmup_scheduler = flow.optimizer.warmup.linear(10, 0.1)
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.01], warmup=warmup_scheduler)
            flow.optimizer.Adam(lr_scheduler).minimize(loss)

            return loss

    """
    def __init__(self, steps, start_multiplier):
        self.steps = steps
        self.start_multiplier = start_multiplier

    @property
    def warmup_conf(self) -> op_conf_pb.WarmupConf:
        warmup_conf = op_conf_pb.WarmupConf()
        warmup_conf.linear_conf.warmup_batches = self.steps
        warmup_conf.linear_conf.start_multiplier = self.start_multiplier
        return warmup_conf


class LrScheduler:
    def __init__(
        self,
        base_lr: Optional[float] = None,
        lr_lbn: Optional[Text] = None,
        warmup: Optional[WarmupConf] = None,
    ):
        self.base_lr = base_lr
        self.lr_lbn = lr_lbn
        self.warmup = warmup

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        raise NotImplementedError()

    def SetLrFieldsInTrainConf(self, train_conf) -> None:
        if self.warmup_conf is not None:
            train_conf.model_update_conf.warmup_conf.CopyFrom(self.warmup_conf)
        if self.lr_lbn is not None:
            assert self.learning_rate_decay_conf is None
            assert self.base_lr is None
            train_conf.primary_lr_lbn = self.lr_lbn
            # primary_lr is a required field
            train_conf.primary_lr = 0
        else:
            assert self.learning_rate_decay_conf is not None
            train_conf.model_update_conf.learning_rate_decay.CopyFrom(
                self.learning_rate_decay_conf
            )
            train_conf.primary_lr = self.base_lr

    @property
    def warmup_conf(self) -> op_conf_pb.WarmupConf:
        if self.warmup is None:
            return None
        return self.warmup.warmup_conf


@oneflow_export("optimizer.CosineScheduler")
class CosineScheduler(LrScheduler):
    r"""This operator creates a Cosine decayed learning rate scheduler. Before the steps specified by user, the learning rate will be updated as: 

    .. math:: 

        & cos\_decay = 0.5*(1+cos(\pi*\frac{current\_batch}{decayed\_batch})) 
        
        & decay\_factor = (1-\alpha)*cos\_decay+\alpha 
        
        & learning\_rate = base\_learning\_rate*decay\_factor

    After the steps specified by user, the learning rate will be :

    .. math:: 

        learning\_rate = {base\_learning\_rate}*{\alpha}  

    Args:
        base_lr (float): The base learning rate (:math:`base\_learning\_rate`)
        steps (int): The decay steps in the scheduler (:math:`decayed\_batch`)
        alpha (float, optional): The learning rate scale factor (:math:`\alpha`). Defaults to 0.0.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            lr_scheduler = flow.optimizer.CosineScheduler(base_lr=0.01,
                                                          steps=10,
                                                          alpha=0.1)
            flow.optimizer.Adam(lr_scheduler).minimize(loss)

            return loss

    """
    def __init__(
        self,
        base_lr: float,
        steps: int,
        alpha: float = 0.0,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.alpha = alpha

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.cosine_conf.decay_batches = self.steps
        learning_rate_decay_conf.cosine_conf.alpha = self.alpha
        return learning_rate_decay_conf


@oneflow_export("optimizer.CustomScheduler")
class CustomScheduler(LrScheduler):
    def __init__(self, lbn: Text):
        super().__init__(lr_lbn=lbn)

    @property
    def learning_rate_decay_conf(self) -> op_conf_pb.LearningRateDecayConf:
        return None


@oneflow_export("optimizer.PiecewiseConstantScheduler")
class PiecewiseConstantScheduler(LrScheduler):
    r"""This operator creates a piecewise constant learning rate scheduler. The change in learning rate can be described as follows:

    .. code-block:: python 

        boundaries = [1000, 2000]
        values = [0.1, 0.01, 0.001]

        if current_step < 1000: 
            learning_rate = 0.1
        elif 1000 < current_step < 2000:
            learning_rate = 0.01
        else:
            learning_rate = 0.001

    Args:
        boundaries (Sequence[int]): A list of train steps. 
        values (Sequence[float]): A list of learning rate values during the different train step boundary. 
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp

        @flow.global_function(type="train")
        def train_job(
                images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler(boundaries=[10, 20],
                                                                     values=[0.1, 0.01, 0.001])
            flow.optimizer.Adam(lr_scheduler).minimize(loss)

            return loss

    """
    def __init__(
        self,
        boundaries: Sequence[int],
        values: Sequence[float],
        warmup: Optional[WarmupConf] = None,
    ):
        assert len(boundaries) + 1 == len(values)
        super().__init__(base_lr=values[0], warmup=warmup)
        self.boundaries = boundaries
        self.values = values

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.piecewise_constant_conf.boundaries.extend(
            self.boundaries
        )
        learning_rate_decay_conf.piecewise_constant_conf.values.extend(self.values)
        return learning_rate_decay_conf


@oneflow_export("optimizer.PiecewiseScalingScheduler")
class PiecewiseScalingScheduler(LrScheduler):
    """This operator creates a piecewise scaled decayed learning rate scheduler. The change in learning rate can be described as follows:

    .. code-block:: python 

        boundaries = [1000, 2000]
        scale = [0.1, 0.01]
        base_lr = 0.1

        if current_step < 1000: 
            learning_rate = base_lr
        elif 1000 < current_step < 2000:
            learning_rate = 0.1*base_lr
        else:
            learning_rate = 0.01*base_lr

    Args:
        base_lr (float): The base learning rate
        boundaries (Sequence[int]): A list of train steps. 
        scale (Union[float, Sequence[float]]): A list of learning rate scaled factor during the different train step boundary. 
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(base_lr=0.1,
                                                                    boundaries=[5, 10],
                                                                    scale=[0.5, 0.1])
            flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)

            return loss

    """
    def __init__(
        self,
        base_lr: float,
        boundaries: Sequence[int],
        scale: Union[float, Sequence[float]],
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.boundaries = boundaries
        if not isinstance(scale, collections.abc.Sequence):
            scale = [scale] * len(boundaries)
        assert len(boundaries) == len(scale)
        self.scale = [1] + list(scale)

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.piecewise_scaling_conf.boundaries.extend(
            self.boundaries
        )
        learning_rate_decay_conf.piecewise_scaling_conf.scales.extend(self.scale)
        return learning_rate_decay_conf


@oneflow_export("optimizer.PolynomialSchduler")
class PolynomialSchduler(LrScheduler):
    r"""This operator creates a polynomial decayed learning rate scheduler. The learning rate will be updated as follow:

    If cycle is `True`, the equation is: 

    .. math:: 

        & decay\_batch = decay\_batch*ceil(\frac{current\_batch}{decay\_batch})  

        & learning\_rate = (base\_lr-end\_lr)*(1-\frac{current\_batch}{decay\_batch})^{pow}+end\_lr

    If cycle is `False`, the equation is:

    .. math:: 

        & decay\_batch = min(decay\_batch, current\_batch)
        
        & learning\_rate = (base\_lr-end\_lr)*(1-\frac{current\_batch}{decay\_batch})^{pow}+end\_lr

    Args:
        base_lr (float): The base learning rate
        steps (int): The decayed steps
        end_learning_rate (float, optional): The final learning rate. Defaults to 0.0001.
        power (float, optional): The power of polynomial. Defaults to 1.0.
        cycle (bool, optional): If cycle is true, the scheduler will decay the learning rate every decay steps. Defaults to False.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

        .. code-block:: python

            import oneflow as flow
            import oneflow.typing as tp

            @flow.global_function(type="train")
            def train_job(
                    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
            ) -> tp.Numpy:
                with flow.scope.placement("gpu", "0:0"):
                    logits = lenet(images, train=True)
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits, name="softmax_loss"
                    )

                lr_scheduler = flow.optimizer.PolynomialSchduler(base_lr=0.001,
                                                                 steps=5,
                                                                 end_learning_rate=0.00001,
                                                                 power=2)
                flow.optimizer.Adam(lr_scheduler).minimize(loss)

                return loss

    """
    def __init__(
        self,
        base_lr: float,
        steps: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.polynomial_conf.decay_batches = self.steps
        learning_rate_decay_conf.polynomial_conf.end_learning_rate = (
            self.end_learning_rate
        )
        learning_rate_decay_conf.polynomial_conf.power = self.power
        learning_rate_decay_conf.polynomial_conf.cycle = self.cycle
        return learning_rate_decay_conf


@oneflow_export("optimizer.LinearCosineScheduler")
class LinearCosineScheduler(LrScheduler):
    r"""This operator creates a linear cosine decayed learning rate scheduler. The learning rate will be updated as follow:

    .. math:: 

        & current\_batch = min(current\_batch, decay\_batch)
        
        & linear\_decay = \frac{(decay\_batch - current\_batch)}{decay\_batch}
        
        & cosine\_decay = 0.5*(1.0+cos(2*\pi*num\_periods*\frac{current\_batch}{decay\_batch}))
        
        & decay\_factor = (\alpha+linear\_decay)*cosine\_decay + \beta 
        
        & learning\_rate = base\_learning\_rate*decay\_factor

    Args:
        base_lr (float): The base learning rate
        steps (int): The decay steps
        num_periods (float, optional): The number of decay periods. Defaults to 0.5.
        alpha (float, optional): The :math:`\alpha` in equation. Defaults to 0.0.
        beta (float, optional): The :math:`\beta` in equation. Defaults to 0.001.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

        .. code-block:: python

            import oneflow as flow
            import oneflow.typing as tp

            @flow.global_function(type="train")
            def train_job(
                    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
            ) -> tp.Numpy:
                with flow.scope.placement("gpu", "0:0"):
                    logits = lenet(images, train=True)
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits, name="softmax_loss"
                    )

                lr_scheduler = flow.optimizer.LinearCosineScheduler(base_lr=0.1,
                                                                    steps=10)
                flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(loss)

                return loss

    """
    def __init__(
        self,
        base_lr: float,
        steps: int,
        num_periods: float = 0.5,
        alpha: float = 0.0,
        beta: float = 0.001,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.num_periods = num_periods
        self.alpha = alpha
        self.beta = beta

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.linear_cosine_conf.decay_batches = self.steps
        learning_rate_decay_conf.linear_cosine_conf.num_periods = self.num_periods
        learning_rate_decay_conf.linear_cosine_conf.alpha = self.alpha
        learning_rate_decay_conf.linear_cosine_conf.beta = self.beta
        return learning_rate_decay_conf


@oneflow_export("optimizer.ExponentialScheduler")
class ExponentialScheduler(LrScheduler):
    r"""This operator creates a exponential decayed learning rate scheduler. The learning rate will be updated as follow:

    If stair case is set to False, the equation is: 

    .. math:: 

        & pow = \frac{current\_batch}{decay\_batch} 
        
        & learning\_rate = base\_learning\_rate*decay\_rate^{pow}
    
    If staircase is set to True, the equation is: 

    .. math:: 

        & pow = floor(\frac{current\_batch}{decay\_batch}) 
            
        & learning\_rate = base\_learning\_rate*decay\_rate^{pow}
    
    Args:
        base_lr (float): The base learning rate
        steps (int): The decay steps
        decay_rate (float): The decay rate
        staircase (bool, optional): If staircase is True, the scheduler decay the learning rate at discrete intervals. Defaults to False.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

        .. code-block::python

            import oneflow as flow
            import oneflow.typing as tp

            @flow.global_function(type="train")
            def train_job(
                    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
            ) -> tp.Numpy:
                with flow.scope.placement("gpu", "0:0"):
                    logits = lenet(images, train=True)
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits, name="softmax_loss"
                    )

                lr_scheduler = flow.optimizer.CosineScheduler(base_lr=0.01,
                                                              steps=10,
                                                              alpha=0.1)
                flow.optimizer.Adam(lr_scheduler).minimize(loss)

                return loss

    """
    def __init__(
        self,
        base_lr: float,
        steps: int,
        decay_rate: float,
        staircase=False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.exponential_conf.decay_batches = self.steps
        learning_rate_decay_conf.exponential_conf.decay_rate = self.decay_rate
        learning_rate_decay_conf.exponential_conf.staircase = self.staircase
        return learning_rate_decay_conf


@oneflow_export("optimizer.InverseTimeScheduler")
class InverseTimeScheduler(LrScheduler):
    r"""This operator creates a inverse time decayed learning rate scheduler. The learning rate will be updated as follow:

    If stair case is set to False, the equation is: 

    .. math:: 

        & step\_ratio = \frac{current\_batch}{decay\_batch}

        & learning\_rate = \frac{base\_learning\_rate}{1+decay\_rate*step\_ratio}
    
    If staircase is set to True, the equation is: 

    .. math:: 

        & step\_ratio = \frac{current\_batch}{decay\_batch}

        & learning\_rate = \frac{base\_learning\_rate}{1+floor(decay\_rate*step\_ratio)}
    
    Args:
        base_lr (float): The base learning rate
        steps (int): The decay steps
        decay_rate (float): The decay rate
        staircase (bool, optional): If staircase is True, the scheduler decay the learning rate at discrete intervals. Defaults to False.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

        .. code-block:: python

            import oneflow as flow
            import oneflow.typing as tp

            @flow.global_function(type="train")
            def train_job(
                    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
            ) -> tp.Numpy:
                with flow.scope.placement("gpu", "0:0"):
                    logits = lenet(images, train=True)
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits, name="softmax_loss"
                    )

                lr_scheduler = flow.optimizer.InverseTimeScheduler(base_lr=0.1,
                                                                   steps=5,
                                                                   decay_rate=0.9)
                flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(loss)

                return loss

    """
    def __init__(
        self,
        base_lr: float,
        steps: int,
        decay_rate: float,
        staircase: bool = False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.inverse_time_conf.decay_batches = self.steps
        learning_rate_decay_conf.inverse_time_conf.decay_rate = self.decay_rate
        learning_rate_decay_conf.inverse_time_conf.staircase = self.staircase
        return learning_rate_decay_conf


@oneflow_export("optimizer.NaturalExpScheduler")
class NaturalExpScheduler(LrScheduler):
    r"""This operator creates a natural exponential decayed learning rate scheduler. The learning rate will be updated as follow:
    
    If stair case is set to False, the equation is: 

    .. math:: 

        & step\_ratio = \frac{current\_batch}{decay\_batch}
        
        & learning\_rate = {base\_learning\_rate}*e^{-decay\_rate*step\_ratio}
    
    If staircase is set to True, the equation is: 

    .. math:: 

        & step\_ratio = \frac{current\_batch}{decay\_batch}
        
        & learning\_rate = {base\_learning\_rate}*e^{-decay\_rate*floor(step\_ratio)}
    
    Args:
        base_lr (float): The base learning rate
        steps (int): The decay steps
        decay_rate (float): The decay rate
        staircase (bool, optional): If staircase is True, the scheduler decay the learning rate at discrete intervals. Defaults to False.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

        .. code-block:: python

            import oneflow as flow
            import oneflow.typing as tp

            @flow.global_function(type="train")
            def train_job(
                    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
            ) -> tp.Numpy:
                with flow.scope.placement("gpu", "0:0"):
                    logits = lenet(images, train=True)
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits, name="softmax_loss"
                    )

                lr_scheduler = flow.optimizer.NaturalExpScheduler(base_lr=0.1,
                                                                  steps=10,
                                                                  decay_rate=0.5)
                flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(loss)

                return loss

    """
    def __init__(
        self,
        base_lr: float,
        steps: int,
        decay_rate: float,
        staircase: bool = False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    @property
    def learning_rate_decay_conf(self) -> Optional[op_conf_pb.LearningRateDecayConf]:
        learning_rate_decay_conf = op_conf_pb.LearningRateDecayConf()
        learning_rate_decay_conf.natural_exp_conf.decay_batches = self.steps
        learning_rate_decay_conf.natural_exp_conf.decay_rate = self.decay_rate
        learning_rate_decay_conf.natural_exp_conf.staircase = self.staircase
        return learning_rate_decay_conf


class Optimizer:
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        loss_scale_factor: Optional[int] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        self.lr_scheduler = lr_scheduler
        self.loss_scale_factor = loss_scale_factor
        self.grad_clipping = grad_clipping
        self.train_step_lbn = train_step_lbn

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        raise NotImplementedError()

    @property
    def train_conf(self) -> job_conf_pb.TrainConf:
        train_conf = job_conf_pb.TrainConf()
        self.lr_scheduler.SetLrFieldsInTrainConf(train_conf)
        update_conf = train_conf.model_update_conf
        if self.grad_clipping is not None:
            update_conf.clip_conf.CopyFrom(self.grad_clipping.clip_conf)
        if self.train_step_lbn is not None:
            train_conf.train_step_lbn = self.train_step_lbn
        if self.loss_scale_factor is not None:
            update_conf.loss_scale_factor = self.loss_scale_factor
        self._SetSpecificFieldsInTrainConf(train_conf)
        return train_conf

    def minimize(
        self, loss: Union[Sequence[remote_blob_util.BlobDef], remote_blob_util.BlobDef]
    ) -> None:
        if not isinstance(loss, collections.abc.Sequence):
            loss = [loss]
        c_api_util.CurJobBuildAndInferCtx_SetTrainConf(self.train_conf)
        for x in loss:
            flow.losses.add_loss(x)


@oneflow_export("optimizer.SGD")
class SGD(Optimizer):
    r"""The optimizer of the stochastic gradient descent algorithm. This algorithm takes a random sample's gradient as an approximate estimate of the overall gradient in small batch gradient descent.  

    When the momentum = 0, the equation of parameters updating is: 

    .. math::

        param_{new} = param_{old} - learning\_rate*grad
    
    With momentum, the equation of parameters updating is:

    .. math::

        & V_{t} = \beta*V_{t-1} + learning\_rate*g_t 

        & param_{new} = param_{old} - V_{t}

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        momentum (float, optional): Momentum factor (:math:`\beta`). Defaults to 0.9.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        
        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            # Set Learning rate as 0.1
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            # Set Momentum=0.9 SGD optimizer
            flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(loss)
            
            return loss
    """

    def __init__(
        self,
        lr_scheduler: LrScheduler,
        loss_scale_factor: Optional[float] = None,
        momentum: float = 0.9,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.momentum = momentum

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        if self.momentum == 0:
            train_conf.model_update_conf.naive_conf.SetInParent()
        else:
            train_conf.model_update_conf.momentum_conf.beta = self.momentum


@oneflow_export("optimizer.Adam")
class Adam(Optimizer):
    r"""The optimizer of the Adam algorithm. This algorithm can adjust the learning rate of each parameter dynamically according to the 1st-moment estimates and the 2nd-moment estimates of gradient.
    
    With bias correction, the equation of parameters updating is: 
    
    .. math::

        & V_t = \beta_1*V_{t-1} + (1-\beta_1)*grad 

        & S_t = \beta_2*S_{t-1} + (1-\beta_2)*{grad} \odot {grad} 

        & \hat{V_t} = \frac{V_t}{1-\beta_1^t} 

        & \hat{S_t} = \frac{S_t}{1-\beta_2^t} 

        & \hat{g} = learning\_rate*\frac{\hat{V_t}}{\sqrt{\hat{S_t}}+\epsilon} 

        & param_{new} = param_{old} - \hat{g}

    Without bias correction, the equation of parameters updating is: 
    
    .. math::

        & V_t = \beta_1*V_{t-1} + (1-\beta_1)*grad 

        & S_t = \beta_2*S_{t-1} + (1-\beta_2)*{grad} \odot {grad} 

        & \hat{g} = learning\_rate*\frac{{V_t}}{\sqrt{{S_t}}+\epsilon} 

        & param_{new} = param_{old} - \hat{g}

    More details please refer to `Adam <https://arxiv.org/abs/1412.6980>`_

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        beta1 (float, optional): The exponential weighted average decay rate for the 1st-moment estimates (:math:`\beta_1`). Defaults to 0.9.
        beta2 (float, optional): The exponential weighted average decay rate for the 2rd-moment estimates (:math:`\beta_2`). Defaults to 0.999.
        epsilon ([type], optional): A small float constant value for numerical stability (:math:`\epsilon`). Defaults to 1e-8.
        do_bias_correction (bool, optional): Whether to do the bias correction. Defaults to False.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        
        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            
            # Set learning rate as 0.001
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
            # Set Adam optimizer
            flow.optimizer.Adam(lr_scheduler, do_bias_correction=False).minimize(loss)
            
            return loss
    """

    def __init__(
        self,
        lr_scheduler: LrScheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        do_bias_correction=False,
        loss_scale_factor: Optional[float] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.do_bias_correction = do_bias_correction

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        train_conf.model_update_conf.adam_conf.beta1 = self.beta1
        train_conf.model_update_conf.adam_conf.beta2 = self.beta2
        train_conf.model_update_conf.adam_conf.epsilon = self.epsilon
        train_conf.model_update_conf.adam_conf.do_bias_correction = (
            self.do_bias_correction
        )


@oneflow_export("optimizer.AdamW")
class AdamW(Optimizer):
    r"""The optimizer of the Adam-weight-decay algorithm. If we use L2 regularization, it will be invalid due to the adaptive learning rate in Adam optimizer (More details please refer to `Adam-weight-decay <https://www.fast.ai/2018/07/02/adam-weight-decay/>`_). So we use Adam-weight-decay algorithm to solve this problem. 

    With bias correction, the equation of parameters updating is: 
    
    .. math::

        & V_t = \beta_1*V_{t-1} + (1-\beta_1)*grad 

        & S_t = \beta_2*S_{t-1} + (1-\beta_2)*{grad} \odot {grad} 

        & \hat{V_t} = \frac{V_t}{1-\beta_1^t} 

        & \hat{S_t} = \frac{S_t}{1-\beta_2^t} 

        & \hat{g} = learning\_rate*(\frac{\hat{V_t}}{\sqrt{\hat{S_t}}+\epsilon}+\lambda*param_{old}) 

        & param_{new} = param_{old} - \hat{g}

    Without bias correction, the equation of parameters updating is: 
    
    .. math::

        & V_t = \beta_1*V_{t-1} + (1-\beta_1)*grad 

        & S_t = \beta_2*S_{t-1} + (1-\beta_2)*{grad} \odot {grad} 

        & \hat{g} = learning\_rate*(\frac{{V_t}}{\sqrt{{S_t}}+\epsilon}+\lambda*param_{old}) 

        & param_{new} = param_{old} - \hat{g}

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        beta1 (float, optional): The exponential weighted average decay rate for the 1st-moment estimates (:math:`\beta_1`). Defaults to 0.9.
        beta2 (float, optional): The exponential weighted average decay rate for the 2rd-moment estimates (:math:`\beta_2`). Defaults to 0.999.
        epsilon ([type], optional): A small float constant value for numerical stability (:math:`\epsilon`). Defaults to 1e-8.
        do_bias_correction (bool, optional): Whether to do the bias correction. Defaults to False.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        weight_decay (Optional[float], optional): The weight decay factor (In the equation is :math:`\lambda`). Defaults to None.
        weight_decay_includes (Optional[Union[Sequence[Text], Text]], optional): The name of the model parameters that use weight decay. Defaults to None.
        weight_decay_excludes (Optional[Union[Sequence[Text], Text]], optional): The name of the model parameters that do not use weight decay. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.

    Note:

        Only one of `weight_decay_includes` and `weight_decay_excludes` can be set. If both are None. All the model parameters will use weight decay. 

    For example: 

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        
        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            
            # Set learning rate as 0.001
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
            # Set AdamW optimizer, weight_decay factor is 0.00005
            flow.optimizer.AdamW(lr_scheduler, 
                    do_bias_correction=False, weight_decay=0.00005).minimize(loss)
            
            return loss

    """

    def __init__(
        self,
        lr_scheduler: LrScheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        do_bias_correction=False,
        loss_scale_factor: Optional[float] = None,
        weight_decay: Optional[float] = None,
        weight_decay_includes: Optional[Union[Sequence[Text], Text]] = None,
        weight_decay_excludes: Optional[Union[Sequence[Text], Text]] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.do_bias_correction = do_bias_correction
        self.weight_decay = weight_decay
        if isinstance(weight_decay_includes, str):
            weight_decay_includes = [weight_decay_includes]
        if isinstance(weight_decay_excludes, str):
            weight_decay_excludes = [weight_decay_excludes]
        self.weight_decay_includes = weight_decay_includes
        self.weight_decay_excludes = weight_decay_excludes

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        train_conf.model_update_conf.adam_conf.beta1 = self.beta1
        train_conf.model_update_conf.adam_conf.beta2 = self.beta2
        train_conf.model_update_conf.adam_conf.epsilon = self.epsilon
        train_conf.model_update_conf.adam_conf.do_bias_correction = (
            self.do_bias_correction
        )
        if self.weight_decay is not None:
            train_conf.model_update_conf.weight_decay_conf.weight_decay_rate = (
                self.weight_decay
            )
            assert not (
                self.weight_decay_excludes is not None
                and self.weight_decay_includes is not None
            )
            if self.weight_decay_includes is not None:
                train_conf.model_update_conf.weight_decay_conf.includes.pattern.extend(
                    self.weight_decay_includes
                )
            elif self.weight_decay_excludes is not None:
                train_conf.model_update_conf.weight_decay_conf.excludes.pattern.extend(
                    self.weight_decay_excludes
                )


@oneflow_export("optimizer.RMSProp")
class RMSProp(Optimizer):
    r"""The optimizer of the RMSProp algorithm. This algorithm uses mean squared gradient to adjust the learning rate. 

    The equation of parameters updating is: 
    
    .. math::

        & S_t = \beta_1*S_{t-1} + (1-\beta_1)*grad \odot grad 

        & param_{new} = param_{old} - \frac{learning\_rate}{\sqrt{S_t+\epsilon}} \odot grad

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        decay_rate (float, optional): The decay factor (:math:`\beta_1`). Defaults to 0.99.
        epsilon (float, optional): A small float constant value for numerical stability (:math:`\epsilon`). Defaults to 1e-8.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            # Set learning rate as 0.001
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
            # Set RMSProp optimizer
            flow.optimizer.RMSProp(lr_scheduler).minimize(loss)

            return loss

    """
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        decay_rate: float = 0.99,
        epsilon: float = 1e-8,
        loss_scale_factor: Optional[float] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.decay_rate = decay_rate
        self.epsilon = epsilon

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        train_conf.model_update_conf.rmsprop_conf.decay_rate = self.decay_rate
        train_conf.model_update_conf.rmsprop_conf.epsilon = self.epsilon


@oneflow_export("optimizer.LARS")
class LARS(Optimizer):
    r"""The optimizer of the Lars algorithm. 

    The equation of parameters updating is: 
    
    .. math::

        & local\_learning\_rate = learning\_rate*lars\_coeff*\frac{\lVert{parm_{old}\rVert}}{\epsilon+\lVert{grad\rVert}} 

        & momentum_t = \beta*momentum_{t-1} + local\_learning\_rate*(grad) 

        & param_{new} = param_{old} - momentum_t

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        momentum_beta (float, optional): The momentum factor (:math:`\beta`). Defaults to 0.9.
        epsilon (float, optional): A small float constant value for numerical stability (:math:`\epsilon`). Defaults to 1e-9.
        lars_coefficient (float, optional): The coefficient factor, it defines how much we trust the layer to change its weights (:math:`lars\_coeff`). Defaults to 0.0001.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.
    
    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp

        @flow.global_function(type="train")
        def train_job(
                images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            # Set learning rate as 0.1
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            # Set LARS optimizer, momentum factor is 0.9
            flow.optimizer.LARS(lr_scheduler, momentum_beta=0.9).minimize(loss)

            return loss

    """
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        momentum_beta: float = 0.9,
        epsilon: float = 1e-9,
        lars_coefficient: float = 0.0001,
        loss_scale_factor: Optional[float] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.momentum_beta = momentum_beta
        self.epsilon = epsilon
        self.lars_coefficient = lars_coefficient

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        train_conf.model_update_conf.lars_conf.momentum_beta = self.momentum_beta
        train_conf.model_update_conf.lars_conf.epsilon = self.epsilon
        train_conf.model_update_conf.lars_conf.lars_coefficient = self.lars_coefficient


@oneflow_export("optimizer.LazyAdam")
class LazyAdam(Optimizer):
    r"""
    The optimizer of the LazyAdam algorithm. This algorithm can adjust the learning rate of each parameter dynamically according to the 1st-moment estimates and the 2nd-moment estimates of the gradient. The difference between Adam optimizer and LazyAdam optimizer is that LazyAdam only updates the element that has gradient in the current batch, it is faster than Adam optimizer. 

    .. math::

        & V_t = \beta_1*V_{t-1} + (1-\beta_1)*grad 

        & S_t = \beta_2*S_{t-1} + (1-\beta_2)*{grad} \odot {grad} 

        & \hat{g} = learning\_rate*\frac{{V_t}}{\sqrt{{S_t}}+\epsilon} 

        & param_{new} = param_{old} - \hat{g}

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        beta1 (float, optional): The exponential weighted average decay rate for the 1st-moment estimates (:math:`\beta_1`). Defaults to 0.9.
        beta2 (float, optional): The exponential weighted average decay rate for the 2rd-moment estimates (:math:`\beta_2`). Defaults to 0.999.
        epsilon ([type], optional): A small float constant value for numerical stability (:math:`\epsilon`). Defaults to 1e-8.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.

    For example:

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        
        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            # Set learning rate as 0.001
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
            # Set LazyAdam optimizer
            flow.optimizer.LazyAdam(lr_scheduler).minimize(loss)

            return loss

    """
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        loss_scale_factor: Optional[float] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        super().__init__(
            lr_scheduler, loss_scale_factor, grad_clipping, train_step_lbn,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _SetSpecificFieldsInTrainConf(self, train_conf):
        train_conf.model_update_conf.lazy_adam_conf.beta1 = self.beta1
        train_conf.model_update_conf.lazy_adam_conf.beta2 = self.beta2
        train_conf.model_update_conf.lazy_adam_conf.epsilon = self.epsilon
