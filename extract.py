"""
Functions for retrieving the images and labels from the MNIST binary files dataset.
"""

import json
import numpy as np
import os
from PIL import Image

# Image training data files
directory = "data/"
training_image_filename = "train-images-idx3-ubyte"
training_labels_filename = "train-labels-idx1-ubyte"
testing_image_filename = "t10k-images-idx3-ubyte"
testing_labels_filename = "t10k-labels-idx1-ubyte"

# Big Endian byte storage conversion from bytecode to decimal integer 
### Obsolete since int.from_bytes(bytes, byteorder = "big", signed = False) 
  # does the same thing
def bytes_to_int(byte_data):
    '''byte_count = len(byte_data)
    total = 0
    for i in range(byte_count):
        total += 16 ** (2 * (byte_count - i - 1)) * byte_data[i]'''
    return int.from_bytes(byte_data, byteorder = "big", signed = False)


# This takes the bytes of the ubyte image file and first finds the total number
# of images to extract. 
def get_images_array(image_filename):
    image_bytes = read_bytes(image_filename)
    images = []
    num_images = bytes_to_int(image_bytes[4 : 8])
    rows = bytes_to_int(image_bytes[8 : 12])
    cols = bytes_to_int(image_bytes[12 : 16])
    for i in range(num_images):
        images.append(np.array(bytearray(image_bytes[16 + i * rows * cols : 16 + rows * cols * (i + 1)])) / 255) 
    return images

"""
Determine how to split the training and validation images 
"""
def get_training_and_validation(training_count = 50000):
    if 0 >= training_count or training_count >= 60000:
        raise ValueError("Training image count must be between 0 and 60000.")
    images = get_images_array(directory + training_image_filename)
    labels = get_labels(directory + training_labels_filename)
    training_images = images[0 : training_count] 
    training_labels = labels[0 : training_count]
    validation_images = images[training_count : 60000]
    validation_labels = labels[training_count : 60000]
    return training_images, training_labels, validation_images, validation_labels

def get_testing_images():
    return get_images_array(directory + testing_image_filename), get_labels(directory + testing_labels_filename)

# Get the labels corresponding to the images 
def get_labels(label_filename):
    label_bytes = read_bytes(label_filename)
    num_labels = bytes_to_int(label_bytes[4 : 8])
    labels = [[0 for i in range(10)] for j in range(num_labels)]
    for i in range(num_labels):
        labels[i][label_bytes[8 + i]] = 1
    return labels

def read_bytes(filename):
    with open(filename, "rb") as f:
        file_bytes = f.read()
        return file_bytes

# Save the image pixel arrays to a json-formatted file and the labels
def save_to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)

# Read a json file
def load_from_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data

# Convert byte array to png image array
'''def to_png_array(pixels):
    header = b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a'
    ihdr = b'\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x1c\x00\x00\x00\x1c'
    idat = bytes(len(pixels))
    print(int.from_bytes(idat, byteorder = "big", signed = False))
    iend = b'\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82'
    return header '''

def export_png(filename, data):
    with open(filename, "wb") as f:
        f.write(data)

""" Take an arbitrary png or jpg file and scale it to 28x28 using any means necessary
"""
def scale_image(imagefile):
    with Image.open(imagefile) as img:
        rescaled = img.resize((28, 28))
        return rescaled