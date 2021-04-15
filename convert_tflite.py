import tensorflow as tf
import argparse
from models.tensorflow.vgg_tf import VGG
from models.tensorflow.mobilenet_tf import MobileNetv1
tf.config.experimental.set_visible_devices([], 'GPU')
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint_path", type=str, help ="directory containing saved model")
parser.add_argument("-m", "--model", type=str, choices= ['VGG', "MBN"], help="directory containing saved model")

model_map = { "VGG": VGG, "MBN": MobileNetv1}
def main():

    args = parser.parse_args()

    model = model_map[args.model]()
    model.load_weights(args.checkpoint_path)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(f'{args.model}.tflite', 'wb') as f:
        f.write(tflite_model)




if __name__ == "__main__":
    main()