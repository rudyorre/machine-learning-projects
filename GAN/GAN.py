import time
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from util import *

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam

class GAN():
    def __init__(self, verbose=False, data_path='faces'):
        self.verbose = verbose

        # Generation resolution - Must be square 
        # Training data is also scaled to this.
        # Note GENERATE_RES 4 or higher  
        # will blow Google CoLab's memory and have not
        # been tested extensively.
        self.GENERATE_RES = 1 # Generation resolution factor 
        # (1=32, 2=64, 3=96, 4=128, etc.)
        self.GENERATE_SQUARE = 32 * self.GENERATE_RES # rows/cols (should be square)
        self.IMAGE_CHANNELS = 3

        # Preview image 
        self.PREVIEW_ROWS = 4
        self.PREVIEW_COLS = 7
        self.PREVIEW_MARGIN = 16

        # Size vector to generate images from
        self.SEED_SIZE = 100

        # Configuration
        self.DATA_PATH = os.path.join('.', data_path)
        self.EPOCHS = 100 # 50
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 60000

        if self.verbose:
            print(f"Will generate {self.GENERATE_SQUARE}px square images.")

        self.preprocess()

        image_shape = (self.GENERATE_SQUARE, self.GENERATE_SQUARE, self.IMAGE_CHANNELS)
        self.generator, self.discriminator = self.build_generator(), self.build_discriminator(image_shape)

        self.generator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)

    def preprocess(self):
        # Image set has 11,682 images.  Can take over an hour 
        # for initial preprocessing.
        # Because of this time needed, save a Numpy preprocessed file.
        # Note, that file is large enough to cause problems for 
        # sum versions of Pickle,
        # so Numpy binary files are used.
        GENERATE_SQUARE = self.GENERATE_SQUARE

        training_binary_path = os.path.join(self.DATA_PATH, f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')

        print(f"Looking for file: {training_binary_path}")

        if not os.path.isfile(training_binary_path):
            start = time.time()
            print("Loading training images...")

            self.training_data = []
            faces_path = os.path.join(self.DATA_PATH, 'images')
            for filename in (tqdm(os.listdir(faces_path)) if self.verbose else os.listdir(faces_path)):
                path = os.path.join(faces_path, filename)
                image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
                self.training_data.append(np.asarray(image))
            self.training_data = np.reshape(self.training_data,(-1, GENERATE_SQUARE, GENERATE_SQUARE, self.IMAGE_CHANNELS))
            self.training_data = self.training_data.astype(np.float32)
            self.training_data = self.training_data / 127.5 - 1.


            print("Saving training image binary...")
            np.save(training_binary_path, self.training_data)
            elapsed = time.time() - start
            if self.verbose: print (f'Image preprocess time: {hms_string(elapsed)}')
        else:
            print("Loading previous training pickle...")
            self.training_data = np.load(training_binary_path)

        # Batch and shuffle the data
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.training_data).shuffle(self.BUFFER_SIZE).batch(self.BUFFER_SIZE)
    
    def build_generator(self):
        model = Sequential()

        model.add(Dense(4*4*256,activation="relu",input_dim=self.SEED_SIZE))
        model.add(Reshape((4,4,256)))

        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
    
        # Output resolution, additional upsampling
        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        if self.GENERATE_RES>1:
            model.add(UpSampling2D(size=(self.GENERATE_RES,self.GENERATE_RES)))
            model.add(Conv2D(128,kernel_size=3,padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        # Final CNN layer
        model.add(Conv2D(self.IMAGE_CHANNELS, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        return model


    def build_discriminator(self, image_shape):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        return model

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy()
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy()
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        seed = tf.random.normal([self.BATCH_SIZE, self.SEED_SIZE])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(seed, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return gen_loss,disc_loss

    def train(self):
        dataset = self.train_dataset
        fixed_seed = np.random.normal(0, 1, (self.PREVIEW_ROWS * self.PREVIEW_COLS, self.SEED_SIZE))
        start = time.time()

        for epoch in range(self.EPOCHS):
            epoch_start = time.time()

            gen_loss_list = []
            disc_loss_list = []

            for image_batch in dataset:
                t = self.train_step(image_batch)
                gen_loss_list.append(t[0])
                disc_loss_list.append(t[1])

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            epoch_elapsed = time.time()-epoch_start
            print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss},'f' {hms_string(epoch_elapsed)}')
            self.save_images(epoch,fixed_seed)

        elapsed = time.time() - start
        print (f'Training time: {hms_string(elapsed)}')

    def save_images(self, cnt,noise):
        image_array = np.full(( 
            self.PREVIEW_MARGIN + (self.PREVIEW_ROWS * (self.GENERATE_SQUARE+self.PREVIEW_MARGIN)), 
            self.PREVIEW_MARGIN + (self.PREVIEW_COLS * (self.GENERATE_SQUARE+self.PREVIEW_MARGIN)), self.IMAGE_CHANNELS), 
            255, dtype=np.uint8)
        
        generated_images = self.generator.predict(noise)

        generated_images = 0.5 * generated_images + 0.5

        image_count = 0
        for row in range(self.PREVIEW_ROWS):
            for col in range(self.PREVIEW_COLS):
                r = row * (self.GENERATE_SQUARE+16) + self.PREVIEW_MARGIN
                c = col * (self.GENERATE_SQUARE+16) + self.PREVIEW_MARGIN
                image_array[r:r+self.GENERATE_SQUARE,c:c+self.GENERATE_SQUARE] = generated_images[image_count] * 255
                image_count += 1

                
        output_path = os.path.join(self.DATA_PATH,f'output_{self.GENERATE_SQUARE}_{self.GENERATE_SQUARE}')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        filename = os.path.join(output_path,f"train-{cnt}.png")
        im = Image.fromarray(image_array)
        im.save(filename)