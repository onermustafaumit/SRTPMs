import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init

class DistributionPoolingFilter(nn.Module):
	r"""Applies 'distribution' pooling as described in the paper
	`Weakly Supervised Clustering by Exploiting Unique Class Count`_ .

	We have a mini-batch of data Input: :math:`(B, N, J)`, where :math:`B` is number of bags, 
	:math:`N` is number of instances 
	in a bag, :math:`J` is number of extracted features from each instance in a bag.

	Given a bag :math:`X` with :math:`N` instances in the mini-batch, 
	for each instance :math:`x_i \in X`, we have :math:`J` extracted features 
	:math:`[f_{x_i}^j | f_{x_i}^j \in \mathbb{R}, j=1,2, \cdots, J] =\mathbf{f}_{x_i} \in \mathbb{R}^J`.

	Let :math:`\tilde{p}^j_{X}(v)` be the estimated marginal distribution of :math:`j^{th}` feature.
	Estimated marginal distribution :math:`\tilde{p}^j_{X}(v)` is obtained by using kernel 
	density estimation, which employs a Gaussian kernel with standard deviation :math:`\sigma`:

	.. math::
		\tilde{p}^j_{X}(v) = \frac{1}{N} \sum_{i=1}^{N}\frac{1}{\sqrt{2\pi{\sigma}^2}} e^{-\frac{1}{2{\sigma}^2} \left(v- f_{x_i}^{j}\right)^2} \ \forall_{j=1,2,\cdots,J}


	Then, the estimated marginal distribution :math:`\tilde{p}^j_{X}(v)` is sampled with :math:`M` bins:

	.. math::
		h^j_{X} = \left[ \tilde{p}^j_{X}(v=v_b) \ | \ v_b=\frac{b}{M-1} \text{ and } b=0,1,\cdots,M-1 \right] \ \forall_{j=1,2,\cdots,J}  \text{ where } h^j_{X} \in \mathbb{R}^M

	.. note::
		Sampling of the estimated distribution is done in the interval of :math:`[0,1]`. Hence, 
		for proper operation Input should be in the interval of :math:`[0,1]`.

	Bag level representation:

	.. math::

		\mathbf{h}_{X} = [h^j_{X} \ | \ h^j_{X} \in \mathbb{R}^M, j=1,2,\cdots,J] \in \mathbb{R}^{MJ}


	Finally, Output: :math:`(B, J, M)` consists of bag level representations of the bags in mini-batch.

	Args:
		num_bins: :math:`M`, number of bins to sample estimated marginal distributions.
			Default: 21
		sigma: :math:`\sigma`, standard deviation of Gaussian kernel.
			Default: 0.0167

	Shape:
		- Input: :math:`(B, N, J)`
		- Output: :math:`(B, J, M)`

	Examples::

		>>> input = torch.normal(0.5, 0.1, (1,200,1))
		>>> print(input.size())
		torch.Size([1, 200, 1])
		>>> m = DistributionPoolingFilter(21,0.0167)
		>>> output = m(input)
		>>> print(output.size())
		torch.Size([1, 1, 21])
		>>> print(output)
		tensor([[[7.7092e-36, 8.8147e-22, 1.2893e-11, 2.4122e-05, 5.8514e-03,
		          1.9012e-02, 3.0407e-02, 6.2145e-02, 1.1265e-01, 1.6227e-01,
		          1.9503e-01, 1.8669e-01, 1.1016e-01, 7.9186e-02, 2.5840e-02,
		          1.0619e-02, 1.0948e-04, 2.4347e-10, 7.0687e-20, 2.6271e-33,
		          0.0000e+00]]])

	.. _`Weakly Supervised Clustering by Exploiting Unique Class Count`:
		https://arxiv.org/abs/1906.07647
	"""


	__constants__ = ['num_bins', 'sigma']

	def __init__(self, num_bins=21, sigma=0.0167):
		super(DistributionPoolingFilter, self).__init__()

		self.num_bins = num_bins
		self.sigma = sigma
		self.alfa = 1/math.sqrt(2*math.pi*(sigma**2))
		self.beta = -1/(2*(sigma**2))

		sample_points = torch.linspace(0,1,steps=num_bins, dtype=torch.float32, requires_grad=False)
		self.register_buffer('sample_points', sample_points)


	def extra_repr(self):
		return 'num_bins={}, sigma={}'.format(
			self.num_bins, self.sigma
		)


	def forward(self, data):
		batch_size, num_instances, num_features = data.size()

		sample_points = self.sample_points.repeat(batch_size,num_instances,num_features,1)
		# sample_points.size() --> (batch_size,num_instances,num_features,num_bins)

		data = torch.reshape(data,(batch_size,num_instances,num_features,1))
		# data.size() --> (batch_size,num_instances,num_features,1)

		diff = sample_points - data.repeat(1,1,1,self.num_bins)
		diff_2 = diff**2
		# diff_2.size() --> (batch_size,num_instances,num_features,num_bins)

		result = self.alfa * torch.exp(self.beta*diff_2)
		# result.size() --> (batch_size,num_instances,num_features,num_bins)

		out_unnormalized = torch.sum(result,dim=1)
		# out_unnormalized.size() --> (batch_size,num_features,num_bins)

		norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
		# norm_coeff.size() --> (batch_size,num_features,1)

		out = out_unnormalized / norm_coeff
		# out.size() --> (batch_size,num_features,num_bins)
		
		return out























