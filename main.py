import chainer
import chainer.functions as F
from chainercv.visualizations import vis_image
from chainercv import utils
from ResNet import ResNetCAM

import numpy as np
import matplotlib.pyplot as plt
from imagenet_class_id import CLASS_ID

import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', '-m', type=str, default="model/ResNet-50-model.npz")
	parser.add_argument('--image', '-i', type=str, default="data/cat.jpeg")
	args = parser.parse_args()

	img = utils.read_image(args.image, color=True)
	img = F.expand_dims(img, 0)
	img = F.resize_images(img, (224, 224))
	ax = vis_image(F.squeeze(img, 0).data)

	model = ResNetCAM()
	chainer.serializers.load_npz(args.model, model)

	with chainer.using_config('train', False):
		conv_outputs, preds = model(img)
	resized_conv_outputs = F.resize_images(conv_outputs, (224, 224))
	to_multiply_tensor = model.fc6.W[2].reshape(1,2048)

	heatmap = np.flip(F.tensordot(to_multiply_tensor, resized_conv_outputs).data, 1)


	ax.imshow(heatmap, cmap='jet', alpha=.4)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	plt.title("Predicted: "+CLASS_ID[int(F.argmax(preds).data)])
	plt.savefig("predicted/result.png")
	plt.show()