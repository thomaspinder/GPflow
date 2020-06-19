# Copyright 2016 James Hensman, alexggmatthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from ..base import Parameter
from ..conditionals import conditional
from ..inducing_variables import InducingPoints
from ..kernels import Kernel
from ..likelihoods import Likelihood
from ..mean_functions import MeanFunction
from ..utilities import to_default_float, print_summary
from .model import GPModel, InputData, MeanAndVariance, RegressionData
from .training_mixins import InternalDataTrainingLossMixin, ExternalDataTrainingLossMixin
from .util import data_input_to_tensor, inducingpoint_wrapper


class SGPMC(GPModel, InternalDataTrainingLossMixin):
    r"""
    This is the Sparse Variational GP using MCMC (SGPMC). The key reference is

    ::

      @inproceedings{hensman2015mcmc,
        title={MCMC for Variatinoally Sparse Gaussian Processes},
        author={Hensman, James and Matthews, Alexander G. de G.
                and Filippone, Maurizio and Ghahramani, Zoubin},
        booktitle={Proceedings of NIPS},
        year={2015}
      }

    The latent function values are represented by centered
    (whitened) variables, so

    .. math::
       :nowrap:

       \begin{align}
       \mathbf v & \sim N(0, \mathbf I) \\
       \mathbf u &= \mathbf L\mathbf v
       \end{align}

    with

    .. math::
        \mathbf L \mathbf L^\top = \mathbf K


    """
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
        inducing_variable: Optional[InducingPoints] = None,
    ):
        """
        data is a tuple of X, Y with X, a data matrix, size [N, D] and Y, a data matrix, size [N, R]
        Z is a data matrix, of inducing inputs, size [M, D]
        kernel, likelihood, mean_function are appropriate GPflow objects
        """
        if num_latent_gps is None:
            num_latent_gps = self.calc_num_latent_gps_from_data(
                data, kernel, likelihood)
        super().__init__(kernel,
                         likelihood,
                         mean_function,
                         num_latent_gps=num_latent_gps)
        self.data = data_input_to_tensor(data)
        self.num_data = data[0].shape[0]
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)
        self.V = Parameter(
            np.zeros((len(self.inducing_variable), self.num_latent_gps)))
        self.V.prior = tfp.distributions.Normal(loc=to_default_float(0.0),
                                                scale=to_default_float(1.0))

    def log_posterior_density(self) -> tf.Tensor:
        return self.log_likelihood_lower_bound() + self.log_prior_density()

    def _training_loss(self) -> tf.Tensor:
        return -self.log_posterior_density()

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_likelihood_lower_bound()

    def log_likelihood_lower_bound(self) -> tf.Tensor:
        """
        This function computes the optimal density for v, q*(v), up to a constant
        """
        # get the (marginals of) q(f): exactly predicting!
        X_data, Y_data = self.data
        fmean, fvar = self.predict_f(X_data, full_cov=False)
        return tf.reduce_sum(
            self.likelihood.variational_expectations(fmean, fvar, Y_data))

    def predict_f(self,
                  Xnew: InputData,
                  full_cov=False,
                  full_output_cov=False) -> MeanAndVariance:
        """
        Xnew is a data matrix of the points at which we want to predict

        This method computes

            p(F* | (U=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at Z,

        """
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            self.V,
            full_cov=full_cov,
            q_sqrt=None,
            white=True,
            full_output_cov=full_output_cov,
        )
        return mu + self.mean_function(Xnew), var


class SSGPMC(GPModel, ExternalDataTrainingLossMixin):
    """
    A stochastic form of the Sparse Variational GP using MCMC (SGPMC). The whitened covariance matrix is still used,
    however, the log-posterior's score function is no longer evaluated for all data points.
    """
    def __init__(self,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 mean_function: Optional[MeanFunction] = None,
                 num_latent_gps: Optional[int] = 1,
                 inducing_variable: Optional[InducingPoints] = None,
                 *,
                 whiten: bool = True,
                 num_data=None):
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.whiten = whiten
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)
        self.V = Parameter(
            np.zeros((len(self.inducing_variable), self.num_latent_gps)))
        self.V.prior = tfp.distributions.Normal(loc=to_default_float(0.0),
                                                scale=to_default_float(1.0))
        # init variational parameters

    def log_posterior_density(self, data: RegressionData) -> tf.Tensor:
        return self.log_likelihood_lower_bound(data) + self.log_prior_density()

    def _training_loss(self, data) -> tf.Tensor:
        return -self.log_posterior_density(data)

    def maximum_log_likelihood_objective(self,
                                         data: RegressionData) -> tf.Tensor:
        return self.log_likelihood_lower_bound(data)

    def log_likelihood_lower_bound(self, data: RegressionData) -> tf.Tensor:
        """
        This function computes the optimal density for v, q*(v), up to a constant
        """
        # get the (marginals of) q(f): exactly predicting!
        X, Y = data
        fmean, fvar = self.predict_f(X, full_cov=False)
        grad_log_pi = tf.reduce_sum(
            self.likelihood.variational_expectations(fmean, fvar, Y))

        # Scale for minibatches
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, grad_log_pi.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], grad_log_pi.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, grad_log_pi.dtype)
        return grad_log_pi * scale

    def predict_f(self,
                  Xnew: InputData,
                  full_cov=False,
                  full_output_cov=False) -> MeanAndVariance:
        """
        Xnew is a data matrix of the points at which we want to predict

        This method computes

            p(F* | (U=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at Z,

        """
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            self.V,
            full_cov=full_cov,
            q_sqrt=None,
            white=True,
            full_output_cov=full_output_cov,
        )
        return mu + self.mean_function(Xnew), var


class SGLD:
    """
    Class implementing regular SGLD as in Algorithm 1 of Nemeth and Fearnhead: https://arxiv.org/pdf/1907.06986.pdf with a decaying learning rate
    """
    def __init__(self, model: SSGPMC, data: tf.data.Dataset):
        self.model = model
        self.data = data
        self.N = self.model.num_data
        self.samples = []
        self.history = []
        self._get_theta()
        # self.param_space = self.determine_sizes(self.model)
        # print(self.param_space)

    def _get_theta(self):
        self.theta = self.model.trainable_parameters
    #
    # @staticmethod
    # def determine_sizes(model: GPModel):
    #     sizes = []
    #     for p in model.trainable_variables:
    #         size = p.shape
    #         sizes.append(size)
    #     return sizes

    def run(self, n_iter: int, stepsize: float = 0.1, batch_size: int = 32, epsilon_decay: bool = False):
        training_iterator = iter(self.data.batch(batch_size))
        # opt = tf.optimizers.SGD(learning_rate=stepsize, momentum=0.0, nesterov=False)
        opt = tf.optimizers.Adam(learning_rate=stepsize)
        samples = []
        # Begin SGLD
        for nit in range(1, n_iter):
            # Load the batch of data S_n
            data_batch = next(training_iterator)
            # Decay the learning rate if required
            if epsilon_decay:
                epsilon_t = stepsize / nit
            else:
                epsilon_t = stepsize
            # Simulate a vector of independent zero-mean Gaussians with variance = I*epsilon
            etas = [tfd.Normal(tf.zeros_like(p), np.sqrt(epsilon_t)).sample() for p in self.model.trainable_variables]

            # Evaluate derivative of the posterior. This is rescaled within the log_posterior_density function
            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                log_target = -self.model.log_posterior_density(data_batch)

            grad_log_pi = tape.gradient(log_target, self.model.trainable_variables)

            delta = [0.5 * epsilon_t * grad + eta for grad, eta in zip(grad_log_pi, etas)]
            # print('-' * 80)
            # print(delta)
            # print('-'*80)
            # print_summary(self.model)
            # print('-' * 80)
            opt.apply_gradients(zip(delta, self.model.trainable_variables))
            # for d, p in zip(delta, self.model.trainable_parameters):
            #     if p.transform:
            #         update = p.unconstrained_variable.numpy() + d
            #         print(update)
            #         p.assign(update)
            #         # p.unconstrained_variable.assign(update)
            #     else:
            #         update = p.numpy() + d
            #         print(update)
            #         p.assign(update)
            # print_summary(self.model)
            # print('-' * 80)
            samples.append([p.numpy() for p in self.model.trainable_variables])
            self.history.append(-self.model.training_loss(data_batch).numpy())
        return samples