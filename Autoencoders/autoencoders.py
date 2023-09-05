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
import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0],True)

tf.random.set_seed(int(time.time()))
np.random.seed(int(time.time()))

data = []
labels = []
dataset_path = 'Vegetable'
classes = os.listdir(dataset_path)
img_rows, img_cols = 28,28


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

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE,self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                Input(shape=(img_rows,img_cols,3)),
                Conv2D(32, kernel_size=3, strides=(2,2), activation='relu'),
                Conv2D(64, kernel_size=3, strides=(2,2), activation='relu'),
                Flatten(),
                Dense(latent_dim + latent_dim)
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Input(shape=(latent_dim,)),
                Dense(units=7*7*64, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 64)),
                Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
                Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same')
            ]
        )
        
    def call(self, inputs):
        mean, log_var = self.encode(inputs)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        return reconstructed
    
    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def encode(self,x):
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, log_var
    
    def reparameterize(self, mean,log_var):
        eps = tf.random.normal(shape=mean.shape)
        return eps*tf.exp(log_var*.5) + mean
    
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100,self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
optimizer = tf.keras.optimizers.Adam(0.001)

    
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_recon = model.decode(z)
    
    recon_loss = tf.reduce_mean(tf.square(x - x_recon))
    
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))
    
    return recon_loss, kl_loss

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        recon_loss, kl_loss = compute_loss(model, x)
        total_loss = recon_loss + kl_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, recon_loss, kl_loss 
    
epochs = 50
batch_size = 64
loss_values = []
vae = VAE(latent_dim=50)

# Training
for epoch in range(epochs):
    print(f"Starting epoch {epoch}")
    for train_x in x_train:
        total_loss, recon_loss, kl_loss = train_step(vae, tf.expand_dims(train_x,0), optimizer)
    loss_values.append(recon_loss.numpy())
    print(f"Finished epoch {epoch}") 

plt.plot(loss_values)
plt.title('Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

vae.save('best_model', save_format='tf')
reconstructed_images = vae.decode(vae.encode(x_test)[0])


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("Original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_images[i].numpy())
    plt.title("Reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
