import tensorflow as tf
import argparse
from tensorflow.keras import datasets
from models.tensorflow.vgg_tf import VGG
import numpy as np
import time
# Argument parser
parser = argparse.ArgumentParser(description='EE379K HW3 - Starter TensorFlow code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch to train')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size

random_seed = 1
tf.random.set_seed(random_seed)

# TODO: Insert your model here
model = VGG()

# TODO: Load the training and testing datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
model.summary()
# TODO: Convert the datasets to contain only float values
train_images = train_images.astype("float32")
test_images = test_images.astype("float32")
# TODO: Normalize the datasets
train_images = train_images/255.0
test_images = test_images/255.0
# TODO: Encode the labels into one-hot format
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
# TODO: Configures the model for training using compile method
start = time.time()
model.compile(optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# TODO: Train the model using fit method
history = model.fit(train_images, train_labels, epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(test_images, test_labels))
total_time = time.time() - start
json.dump(history.history, open("tf_history_1.json", 'w'))
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'loss:{test_loss} acc:{test_acc}, time(s): {total_time}')
# TODO: Save the weights of the model in .ckpt format
model.save_weights("vgg1_tf.ckpt")
