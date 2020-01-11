import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from convert import pytorch_to_caffe
from convert import mgn_res
from zhaohui import PED_CLA_004
from PEDEXT008 import mcc
from videoReid import video_reid
from abnerReid import mgn

use_gpu = torch.cuda.is_available()

def videoBase_reid():
	name="video reid"
	model_path = './videoReid/video_reid.pth.tar'
	data_path = './videoReid/data/'
	
	extractor = video_reid.videoFeatureExtractor(model_path)
	net = extractor.feature_model()
	
	net.eval()
	imgs = Variable(torch.rand([4, 4, 3, 224, 112]))

	device = torch.device('cuda' if use_gpu else 'cpu')
	imgs = extractor.video2clips(data_path)
	with torch.no_grad():
		imgs = imgs.to(device)
		b, n, s, c, h, w = imgs.size()
		assert(b==1)
		imgs = imgs.view(b*n, s, c, h, w)
		print(imgs.size())
		#features, weights = self.model(imgs)
	print("net----->", net)
	pytorch_to_caffe.trans_net(net, imgs, name)
	pytorch_to_caffe.save_prototxt('videoreid2.prototxt')
	pytorch_to_caffe.save_caffemodel('videoreid2.caffemodel')
	


def reid008():
	name="MCC"
	# load model
	net = mcc.MCC()
	#net = PED_CLA_004.MyResNet50()
	print(net)

	checkpoint = torch.load('PEDEXT008/PED_EXT_008.pt')
	net.load_state_dict(checkpoint)
	net.eval()

	input_var = Variable(torch.rand([1, 3, 384, 128]))
	#input_var = Variable(torch.ones([1, 3, 224, 224]))
	# output_var = net.forward(input_var)

	# convert
	pytorch_to_caffe.trans_net(net, input_var, name)
	pytorch_to_caffe.save_prototxt('reid008.prototxt')
	pytorch_to_caffe.save_caffemodel('reid008.caffemodel')
	
	print("Done!")
	
def abnerReid():
	name="CMGN"
	device = torch.device('cuda' if use_gpu else 'cpu')
	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# load model
	net = mgn.CMGN()
	net = net.to(device)
	net.eval()
	#net = PED_CLA_004.MyResNet50()
	print(net)

	checkpoint = torch.load('/home/hup/project/ReID/thd_sample_abner/best_thd-p112-t1000-c25.pth.tar')
	net.load_state_dict(checkpoint['state_dict'], strict=False)
	net.eval()

	example = Image.open('/home/hup/project/ReID/thd_sample_abner/test.jpg')
	# Preprocess image
	tfms = transforms.Compose([transforms.Resize((384, 128)),
							   transforms.ToTensor(),
							   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	example = tfms(example).unsqueeze(0).to(device)
	#input_var = Variable(torch.rand([1, 3, 384, 128]))
	#input_var = Variable(torch.ones([1, 3, 224, 224]))
	# output_var = net.forward(input_var)

	# convert
	pytorch_to_caffe.trans_net(net, example, name)
	pytorch_to_caffe.save_prototxt('abnerReid.prototxt')
	pytorch_to_caffe.save_caffemodel('abnerReid.caffemodel')
	
	print("Done!")

if __name__=='__main__':
	abnerReid()
	