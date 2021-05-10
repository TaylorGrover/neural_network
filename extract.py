import json
import numpy as np
import os

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
                images.append(np.array(bytearray(image_bytes[16 + i * rows * cols : 16 + rows * cols * (i + 1)])))
        return images

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
