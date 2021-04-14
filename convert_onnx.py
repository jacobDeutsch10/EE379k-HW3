import torch
import keras2onnx
from models.pytorch.mobilenet_pt import MobileNetv1 as MBN_pt
from models.tensorflow.mobilenet_tf import MobileNetv1 as MBN_tf
from models.pytorch.vgg_pt import VGG as VGG_pt
from models.tensorflow.vgg_tf import VGG as VGG_tf

#MBN_pt
model = MBN_pt()
model.load_state_dict(torch.load("mbnv1_pt.pt"))
torch.onnx.export(model,torch.rand(1,3,32,32), f"mbnv1_pt.onnx",export_params=True,opset_version=10)

#MBN_tf
model = MBN_tf()
model.load_weights("mbnv1_tf.ckpt")
onnx_model = keras2onnx.convert_keras(model,model.name)
keras2onnx.save_model(onnx_model,f"mbnv1_tf.onnx")

#VGG_pt
model = torch.load("vgg11_1.pt")
model.cuda()
torch.onnx.export(model,torch.rand(1,3,32,32), f"vgg_pt.onnx",export_params=True,opset_version=10)

#VGG_tf
model = VGG_tf()
model.load_weights("vgg1_tf.ckpt")
onnx_model = keras2onnx.convert_keras(model,model.name)
keras2onnx.save_model(onnx_model,f"vgg.onnx")