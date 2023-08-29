from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import Lambda,Input, Conv2D, Dense, BatchNormalization, Flatten, LeakyReLU, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K



physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0],True)

data = []
labels = []
dataset_path = 'Vegetable'
classes = os.listdir(dataset_path)
img_rows, img_cols = 40,40


for class_id, class_name in enumerate(classes):
    folder_path = os.path.join(dataset_path, class_name)
    image_files = os.listdir(folder_path)
    print(class_id)
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = image.load_img(image_path,target_size=(img_rows, img_cols,3),color_mode='rgb')
        img_array = image.img_to_array(img)
        data.append(img_array)

X = np.array(data)

x_train, x_temp = train_test_split(X, test_size=0.2, random_state=int(time.time()))
x_test,x_val = train_test_split(x_temp, test_size=0.5, random_state=int(time.time()))

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3).astype('float32')/255.0
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3).astype('float32')/255.0
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 3).astype('float32')/255.0

def encoder(input_encoder):
    inputs = Input(shape=input_encoder)
    
    x = Conv2D(32, kernel_size=3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(64, kernel_size=3,strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(64, kernel_size=3,strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(64, kernel_size=3,strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(64, kernel_size=3,strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    flatten = Flatten()(x)
    mean = Dense(200)(flatten)
    log_var = Dense(200)(flatten)
    model = Model(inputs, (mean,log_var))
    return model

def decoder(input_decoder):
     
    inputs = Input(shape=input_decoder, name='input_layer')
    x = Dense(4096, name='dense_1')(inputs)
    x = Reshape((8,8,64), name='Reshape')(x)
     
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(name='bn_1')(x)
    x = LeakyReLU(name='lrelu_1')(x)
    
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = LeakyReLU(name='lrelu_2')(x)
  
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = LeakyReLU(name='lrelu_3')(x)
    
    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(name='bn_4')(x)
    x = LeakyReLU(name='lrelu_4')(x)
 
    outputs = Conv2DTranspose(3, kernel_size=3, strides=2,padding='same', activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

def sampling_reparameterization(args):
    mean, log_var = args
    epsilon = tf.random.normal(shape=tf.shape(mean))
    return mean * tf.exp(log_var/2)*epsilon

def sampling(input_1, input_2):
    mean = Input(shape=input_1)
    log_var = Input(shape=input_2)
    out = Lambda(sampling_reparameterization)([mean,log_var])
    enc = Model([mean,log_var], out)
    return enc

input_shape = (img_rows,img_cols,3)
def vae_loss(x, reconstructed_x):
    reconstruction_loss = tf.keras.losses.mse(K.flatten(x), K.flatten(reconstructed_x))
    reconstruction_loss *= img_rows * img_cols * 3
    kl_loss = 1 + log_var - K.square(mean) - K.exp(log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(reconstruction_loss + kl_loss)

# Build the VAE model
encoder_model = encoder((img_rows, img_cols, 3))
decoder_model = decoder(200)

inputs = Input(shape=(img_rows, img_cols, 3))
mean, log_var = encoder_model(inputs)
sampling_model = sampling(200, 200)
z = sampling_model([mean, log_var])
outputs = decoder_model(z)

vae = Model(inputs, outputs)
vae.add_loss(vae_loss(inputs, outputs))
vae.compile(optimizer='adam')

# Early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint("best_vae_model.h5", save_best_only=True)

# Train VAE
history = vae.fit(x_train, x_train,
                  epochs=100,
                  batch_size=32,
                  validation_data=(x_val, x_val),
                  callbacks=[early_stopping, checkpoint])

# Plotting the reconstruction loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Reconstruct images from the test set
reconstructed_imgs = vae.predict(x_test)

# Display original and reconstructed images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(img_rows, img_cols, 3))
    ax.axis('off')

    # Reconstructed images
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(reconstructed_imgs[i].reshape(img_rows, img_cols, 3))
    ax.axis('off')
plt.show()