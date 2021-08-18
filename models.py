import tensorflow as tf
import tensorflow.keras.layers as layer
from tensorflow.keras import Model

class Encoder(Model):
  def __init__(self, encoded_space_dim):
    super(Encoder, self).__init__()
    self.drop = layer.Dropout(rate=0.6)
    self.conv1 = layer.Conv1D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')
    self.conv2 = layer.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same')
    self.conv3 = layer.Conv1D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same')
    self.conv4 = layer.Conv1D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')
    self.flatten = layer.Flatten()
    self.d2 = layer.Dense(encoded_space_dim, activation='relu')
  
  def call(self, x):
    #x = self.drop(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.flatten(x)
    #x = self.d1(x)
    x = self.d2(x)
    return x

class Decoder(Model):
  def __init__(self):
    super(Decoder, self).__init__()
    self.d1 = layer.Dense(8 * 32, activation='relu')
    self.reshape = layer.Reshape((8, 32))
    self.tconv1 = layer.Conv1DTranspose(filters=64, kernel_size=3, strides=2, activation='relu', padding='same')
    self.tconv2 = layer.Conv1DTranspose(filters=128, kernel_size=3, strides=2, activation='relu', padding='same')
    self.tconv3 = layer.Conv1DTranspose(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')
    self.tconv4 = layer.Conv1DTranspose(filters=6, kernel_size=3, strides=2, activation='sigmoid', padding='same')

  def call(self, x):
    x = self.d1(x)
    x = self.reshape(x)
    x = self.tconv1(x)
    x = self.tconv2(x)
    x = self.tconv3(x)
    x = self.tconv4(x)

    return x

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.models.Sequential(
        [
            layer.InputLayer(input_shape=(128, 6)),
            layer.Conv1D(filters=256, kernel_size=7, strides=2, activation='relu', padding='same'),
            layer.Conv1D(filters=126, kernel_size=5, strides=2, activation='relu', padding='same'),
            layer.Conv1D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Conv1D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Flatten(),
            # No activation
            layer.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.models.Sequential(
        [
            layer.InputLayer(input_shape=(latent_dim)),
            layer.Dense(units=8*32, activation='relu'),
            layer.Reshape((8, 32)),
            layer.Conv1DTranspose(filters=64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Conv1DTranspose(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Conv1DTranspose(filters=256, kernel_size=5, strides=2, activation='relu', padding='same'),
            layer.Conv1DTranspose(filters=6, kernel_size=7, strides=2, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=False)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
  
  def call(self, x):
    mean, log_var = self.encode(x)
    codings = self.reparameterize(mean, log_var)
    return self.decode(codings)

def ConvolutionalClassificationNN(num_labels):
    
    return tf.keras.models.Sequential(
        [
          layer.Conv1D(filters=256, kernel_size=9, strides=1, activation='relu', padding='same'), #best model kernel size 9
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
         
          layer.Conv1D(filters=128, kernel_size=7, strides=1, activation='relu', padding='same'), #best model kernel size 7
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
          
          layer.Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='same'), #best model kernel size 5
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
          
          layer.Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'), #best model kernel size 3
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
         
          layer.Conv1D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same'), #best model kernel size 3
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
         
          layer.Flatten(),
          layer.Dropout(0.4),
          layer.Dense(num_labels, activation='softmax'),
        ]
    )

def RNNAutoEncoder(latent_space):
    
    return tf.keras.models.Sequential(
        [
          layer.LSTM(4, activation='relu', input_shape=(128, 6), return_sequences=True),
          layer.LSTM(2, activation='relu', return_sequences=True),
          layer.LSTM(6, activation='relu'),
          layer.RepeatVector(128),
          layer.LSTM(4, activation='relu', return_sequences=True),
          layer.LSTM(2, activation='relu', return_sequences=True),
          layer.TimeDistributed(layer.Dense(6))

        ]
    )