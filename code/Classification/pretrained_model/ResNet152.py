import torch
import torchvision.models as models


def get_ResNet152_unpre(classes):
	net = models.resnet152(pretrained=False)
	net.fc = torch.nn.Linear(net.fc.in_features, classes)

	return net


def get_ResNet152_pre(classes):
	net = models.resnet152(pretrained=True)
	net.fc = torch.nn.Linear(net.fc.in_features, classes)

	return net
