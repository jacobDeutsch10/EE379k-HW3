import numpy as np
import onnxruntime
from tqdm import tqdm
import os
from PIL import Image
import argparse
import sysfs_paths as sysfs
import telnetlib as tel
import psutil
from statistics import mean


# TODO: add argument parser
parser = argparse.ArgumentParser(description='EE379K HW3 - Deployment code')
# TODO: add one argument for selecting PyTorch or TensorFlow option of the code
parser.add_argument('--framework', type=str, default='pytorch', help='Pytorch or Tensorflow')
# TODO: add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument('--model', type=str, default='VGG', help='VGG or MBNV1')
# TODO: Modify the rest of the code to use those arguments correspondingly

 # TODO: insert ONNX model name
args = parser.parse_args()
if args.framework == 'pytorch':
    if args.model == 'VGG':
        onnx_model_name = "vgg_pt.onnx"
    else:
        onnx_model_name = "mbnv1_pt.onnx"
elif args.framework == 'tensorflow':
    if args.model == 'VGG':
        onnx_model_name = "vgg.onnx"
    else:
        onnx_model_name = "mbnv1_tf.onnx"

# Create Inference session using ONNX runtime
sess = onnxruntime.InferenceSession(onnx_model_name)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation used for PyTorch models
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

def getTelnetPower(SP2_tel, last_power):    #for both devices
    tel_dat = str(SP2_tel.read_very_eager())
    print("telnet reading:", tel_dat)
    findex = tel_dat.rfind('\n')
    findex2 = tel_dat[:findex].rfind('\n')
    findex2 = findex2 if findex2 != -1 else 0
    ln = tel_dat[findex2: findex].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power

def getTemps():         #for odroids
    temp1 = []
    for i in range(4):
        temp =  float(file(sysfs.fn_thermal_sensor.format(i), 'r').readline().strip())/1000
        temp1.append(temp)
    t1=temp1[3]
    temp1[3] = temp1[1]
    temp1[1] = t1
    return temp1

SP2_tel = tel.Telnet('192.168.4.1')
out_fname = "problem3_readings.txt"
header = "power temperature"
header = "\t".joing(header.split(" "))
out_file = open(out_fname,'w')
out_file.write(header)
out_file.write('\n')
power = 0.0
# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
accuracy = 0
# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("HW3_files/test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("HW3_files/test_deployment", filename)).resize((32, 32)) as img:
        print("Image shape:", np.float32(img).shape)

        if args.framework == 'pytorch' or (args.framework == 'tensorflow'and args.model =='VGG'):
        # For PyTorch models ONLY: normalize image
            input_image = (np.float32(img) / 255. - mean) / std
        # For PyTorch models ONLY: Add the Batch axis in the data Tensor (C, H, W)
            input_image = np.expand_dims(np.float32(input_image), axis=0)

        if args.framework == 'tensorflow':
        # For TensorFlow models ONLY: Add the Batch axis in the data Tensor (H, W, C)
            input_image = np.expand_dims(np.float32(img), axis=0)
        print("Image shape after expanding size:", input_image.shape)

        if args.framework == 'pytorch':
        # For PyTorch models ONLY: change the order from (B, H, W, C) to (B, C, H, W)
            input_image = input_image.transpose([0, 3, 1, 2])

        # Run inference and get the prediction for the input image
        pred_onnx = sess.run(None, {input_name: input_image})[0]

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]
        #true label
        true_class = ''.join([i for i in filename[:-4] if i.isalpha()])
        if pred_class == true_class:
            accuracy += 1
        power = getTelnetPower(SP2_tel,power)

        temps = getTemps()
        temperature = mean(temps)
        fmt_str = '{}\t'*2
        out = fmt_str.format(power,temperature)
        out_file.write(out)
        out_file.write('\n')
print("Accuracy: " + str(accuracy/10000))