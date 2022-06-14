import tensorflow as tf
import numpy as np
from PIL import Image
# from tqdm import tqdm
import os
import time
import argparse
import matplotlib.pyplot as plt

# Custom GAN
from GAN import GAN

def main(
    verbose=False
):
    # image = Image.open(os.path.join('.', 'faces', 'images', '9326871.1.jpg'))
    # plt.imshow(image)
    # plt.show()

    model = GAN(verbose=verbose)

    # generator = model.build_generator()

    # noise = tf.random.normal([1, model.SEED_SIZE])
    # generated_image = generator(noise, training=False)

    # plt.imshow(generated_image[0, :, :, 0])
    # plt.show()
    model.train()

    model.generator.save(os.path.join(model.DATA_PATH, 'face_generator.h5'))

    print('Finished')

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true', help='print training progress')

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))