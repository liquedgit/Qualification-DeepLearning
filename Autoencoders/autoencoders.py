from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import Lambda,Input, Conv2D, Dense, BatchNormalization, Flatten, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0],True)

tf.random.set_seed(int(time.time()))
np.random.seed(int(time.time()))

data = []
labels = []
dataset_path = 'Vegetable'
classes = os.listdir(dataset_path)
img_rows, img_cols = 35, 35
recons_loss_list = []
kl_div_list = []
total_loss_list = []

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


learning_rate = 0.001
batch_size = 256
epochs = 50
latent_dim = 512
hidden_dim =1024
image_size = img_rows * img_cols * 3
np.random.seed(25)
tf.random.set_seed(25)

# print( x_train.shape[1])
# print(x_train.shape[2])
# print(image_size)



class VAE(tf.keras.Model):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        h_dim = dim[0]
        z_dim = dim[1]
        self.fc1 = Dense(h_dim, activation='relu', kernel_regularizer=l2(0.001))
        self.fc2 = Dense(h_dim//2, activation='relu', kernel_regularizer=l2(0.001))
        self.fc3 = Dense(z_dim, kernel_regularizer=l2(0.001))
        self.fc4 = Dense(z_dim, kernel_regularizer=l2(0.001))
        self.fc5 = Dense(h_dim//2, activation='relu', kernel_regularizer=l2(0.001))
        self.fc6 = Dense(h_dim, activation='relu', kernel_regularizer=l2(0.001))
        self.fc7 = Dense(image_size, kernel_regularizer=l2(0.001))
    
    def encode(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        return self.fc3(h), self.fc4(h)
    
    def reparameterize(self, mu, log_var):
        std = tf.exp(log_var * 0.5)
        eps = tf.random.normal(std.shape)
        return mu + eps * std
    
    def decode_logits(self, z):
        h = self.fc5(z)
        h = self.fc6(h)
        return self.fc7(h)
    
    def decode(self,z):
        return tf.nn.sigmoid(self.decode_logits(z))
    
    def call(self, inputs, training=None, mask=None):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        x_recons = self.decode_logits(z)
        return x_recons, mu, log_var

model = VAE([hidden_dim, latent_dim])
model.build(input_shape=(4, image_size))

print(model.summary())
optimizer = tf.keras.optimizers.Adam(learning_rate)

num_batches = x_train.shape[0] // batch_size
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(batch_size).batch(batch_size)
for epoch in range(epochs):
    for step, x in enumerate(train_dataset):
        x = tf.reshape(x, [-1, img_rows, img_cols, 3])
        x_flat = tf.reshape(x, [-1, image_size])

        with tf.GradientTape() as tape:
            x_recons, mu, log_var = model(x_flat)
            x_recons = tf.reshape(x_recons, [-1, img_rows, img_cols, 3])

            
            kl_div = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var) + 1e-10, axis=-1)
            kl_div = tf.reduce_mean(kl_div)
            recons_loss = tf.reduce_mean(tf.square(x - x_recons))

            loss = recons_loss + kl_div
            recons_loss_list.append(recons_loss.numpy())
            kl_div_list.append(kl_div.numpy())
            total_loss_list.append(loss.numpy())

        gradients = tape.gradient(loss, model.trainable_variables)
        gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if (step+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{num_batches}], Recent Loss : {recons_loss}, KL Div : {kl_div}")


z = tf.random.normal((batch_size, latent_dim))
out = model.decode(z) * 255
out = tf.reshape(out, [-1,img_rows,img_cols]).numpy()
out = out.astype(np.uint8)

model.save_weights('./Autoencoders/best_model.h5')

plt.figure()
plt.plot(recons_loss_list, label='Reconstruction Loss')
plt.plot(kl_div_list, label='KL Divergence')
plt.plot(total_loss_list, label='Total Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Training Steps')
plt.show()


number = 10
plt.figure(figsize=(20,4))
for index in range(number):
    ax = plt.subplot(2, number, index+1)
    plt.xlabel("Rec")
    plt.imshow(x_train[index])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("./Autoencoders/Original image")

number = 10
plt.figure(figsize=(20,4))
for index in range(number):
    ax = plt.subplot(2, number, index+1)
    plt.xlabel("Rec")
    plt.imshow(out[index])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("./Autoencoders/Reconstructed Image")
plt.show()