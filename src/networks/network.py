from .fc import load_fc, load_fc_tf
from .resnet18 import load_resnet18, load_resnet18_tf
from .lenet5 import load_lenet, load_lenet_tf
from .convnet import load_convnet, load_convnet_tf
from .vgg16 import load_vgg16, load_vgg16_tf

networks = {
	'resnet18': 			load_resnet18,
	'fc':					load_fc,
	'lenet':				load_lenet,
	'convnet':				load_convnet,
	'vgg16':				load_vgg16,
}

processors = {
	'resnet18': 			load_resnet18_tf,
	'fc':					load_fc_tf,
	'lenet':				load_lenet_tf,
	'convnet':				load_convnet_tf,
	'vgg16':				load_vgg16_tf,
}