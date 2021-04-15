from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import tflite_runtime.interpreter as tflite
import argparse
import time

# TODO: add argument parser
parser = argparse.ArgumentParser()
# TODO: add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument("-m", "--model", type=str, choices= ['VGG', "MBN"], help="directory containing saved model")
# TODO: Modify the rest of the code to use the arguments correspondingly
model = parser.parse_args().model
tflite_model_name = f"{model}.tflite" # TODO: insert TensorFlow Lite model name

# Get the interpreter for TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=tflite_model_name)

# Very important: allocate tensor memory
interpreter.allocate_tensors()

# Get the position for inserting the input Tensor
input_details = interpreter.get_input_details()
# Get the position for collecting the output prediction
output_details = interpreter.get_output_details()

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
total_pred = 0
true_pred = 0
start_time = time.time()
for filename in tqdm(os.listdir("HW3_files/test_deployment")):
  with Image.open(os.path.join("HW3_files/test_deployment", filename)).resize((32, 32)) as img:
    input_image = np.expand_dims(np.float32(img), axis=0)*1.0/255

    # Set the input tensor as the image
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run the actual inference
    interpreter.invoke()

    # Get the output tensor
    pred_tflite = interpreter.get_tensor(output_details[0]['index'])

    # Find the prediction with the highest probability
    top_prediction = np.argmax(pred_tflite[0])

    # Get the label of the predicted class
    pred_class = label_names[top_prediction]
    true_class = filename.split("_")[1].split(".")[0]
    true_pred += 1 if pred_class == true_class else 0
    total_pred += 1
print(f"accuracy: {100*true_pred/total_pred}, time: {time.time()-start_time}")
