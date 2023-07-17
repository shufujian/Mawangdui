import torch
import torchvision.models as models

def get_DenseNet201_unpre(classes):
	net = models.densenet201(pretrained=False)
	in_features = net.classifier.in_features
	net.classifier = torch.nn.Linear(in_features, classes)

	return net

def get_DenseNet201_pre(classes):
	net = models.densenet201(pretrained=True)
	in_features = net.classifier.in_features
	net.classifier = torch.nn.Linear(in_features, classes)


	return net

