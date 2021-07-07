import math

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from resnet_no_bn import resnet18
from distribution_pooling_filter import DistributionPoolingFilter


class FeatureExtractor(nn.Module):

	def __init__(self, num_features=32):
		super(FeatureExtractor, self).__init__()

		self._model_conv = resnet18()
		
		num_ftrs = self._model_conv.fc.in_features
		self._model_conv.fc = nn.Linear(num_ftrs, num_features)
		# print(self._model_conv)

		# self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()

	def forward(self, x):
		out = self._model_conv(x)
		# out = self.sigmoid(out)
		out = self.relu(out)
		# out = torch.clamp(out, min=0., max=1.)

		return out

class RepresentationTransformation(nn.Module):
	def __init__(self, num_features=32, num_bins=11, num_classes=10):
		super(RepresentationTransformation, self).__init__()

		self.fc = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(num_features * num_bins, 384),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(384, 192),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(192, num_classes)
			)

	def forward(self, x):

		out = self.fc(x)

		return out

class Model(nn.Module):

	def __init__(self, num_classes=10, num_instances=32, num_features=32, num_bins=11, sigma=0.1):
		super(Model, self).__init__()
		self._num_classes = num_classes
		self._num_instances = num_instances
		self._num_features = num_features
		self._num_bins = num_bins
		self._sigma = sigma

		# feature extractor module
		self._feature_extractor = FeatureExtractor(num_features=num_features)

		# MIL pooling filter
		self._mil_pooling_filter = DistributionPoolingFilter(num_bins=num_bins, sigma=sigma)

		# bag-level representation transformation module
		self._representation_transformation = RepresentationTransformation(num_features=num_features, num_bins=num_bins, num_classes=num_classes)


	def forward(self, x):

		out = self._feature_extractor(x)
		out = torch.reshape(out,(-1,self._num_instances,self._num_features))

		out = self._mil_pooling_filter(out)
		out = torch.flatten(out, 1)

		out = self._representation_transformation(out)

		return out






