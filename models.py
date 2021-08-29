import tensorflow as tf
import tensorflow.keras.layers as layer
from tensorflow.keras import Model

def Encoder(encoded_space_dim):
  return tf.keras.models.Sequential([
    layer.Conv1D(filters=32, kernel_size=9, strides=2, activation='relu', padding='same'),
    layer.Conv1D(filters=64, kernel_size=7, strides=2, activation='relu', padding='same'),
    layer.Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', padding='same'),
    layer.Conv1D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same'),
    layer.Flatten(),
    layer.Dropout(rate=0.2),
    layer.Dense(encoded_space_dim),
    #layer.ActivityRegularization(l1=1e-3)
  ])
def Decoder():
  return tf.keras.models.Sequential([
    layer.Dense(8 * 256, activation='relu'),
    layer.Reshape((8, 256)),
    layer.Conv1DTranspose(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'),
    layer.Conv1DTranspose(filters=64, kernel_size=5, strides=2, activation='relu', padding='same'),
    layer.Conv1DTranspose(filters=32, kernel_size=7, strides=2, activation='relu', padding='same'),
    #layer.Conv1DTranspose(filters=16, kernel_size=9, strides=2, activation='relu', padding='same'),
    layer.Conv1DTranspose(filters=6, kernel_size=3, strides=2, padding='same'),
  ])


class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, **kwargs):
    super(CVAE, self).__init__(**kwargs)
    self.latent_dim = latent_dim
    self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
    self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    self.encoder = tf.keras.models.Sequential(
        [
            layer.InputLayer(input_shape=(128, 6)),
            layer.Conv1D(filters=16, kernel_size=9, strides=2, activation='relu', padding='same'),
            layer.Conv1D(filters=32, kernel_size=7, strides=2, activation='relu', padding='same'),
            layer.Conv1D(filters=64, kernel_size=5, strides=2, activation='relu', padding='same'),
            layer.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Flatten(),
            layer.Dropout(rate=0.2),
            layer.Dense(latent_dim + latent_dim),
        ], name="encoder"
    )

    self.decoder = tf.keras.models.Sequential(
        [
            layer.InputLayer(input_shape=(latent_dim)),
            layer.Dense(units=8*512, activation='relu'),
            layer.Reshape((8, 512)),
            layer.Conv1DTranspose(filters=256, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Conv1DTranspose(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Conv1DTranspose(filters=64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Conv1DTranspose(filters=6, kernel_size=3, strides=2, padding='same'),
        ], name="decoder"
    )
  @property
  def metrics(self):
      return [
          self.total_loss_tracker,
          self.reconstruction_loss_tracker,
          self.kl_loss_tracker,
      ]

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=False)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=tf.shape(mean))
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

  def train_step(self, data):
    x, _ = data
    with tf.GradientTape() as tape:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        logits = self.decode(z)
        kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = tf.reduce_mean(kl_loss)
        r_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(x, logits), axis=1)
        r_loss = tf.reduce_mean(r_loss)
        loss = r_loss + kl_loss
        
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.total_loss_tracker.update_state(loss)
    self.reconstruction_loss_tracker.update_state(r_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    return {
        "loss": self.total_loss_tracker.result(),
        "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        "kl_loss": self.kl_loss_tracker.result(),
    }


class LSRAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, **kwargs):
    super(LSRAE, self).__init__(**kwargs)
    self.latent_dim = latent_dim
    self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
    self.kl_loss_tracker = tf.keras.metrics.Mean(name="pred_loss")
    self.encoder = tf.keras.models.Sequential(
        [
            layer.InputLayer(input_shape=(128, 6)),
            layer.Conv1D(filters=16, kernel_size=9, strides=2, activation='relu', padding='same'),
            layer.Conv1D(filters=32, kernel_size=7, strides=2, activation='relu', padding='same'),
            layer.Conv1D(filters=64, kernel_size=5, strides=2, activation='relu', padding='same'),
            layer.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Flatten(),
            layer.Dropout(rate=0.2),
            layer.Dense(latent_dim),
        ], name="encoder"
    )

    self.decoder1 = tf.keras.models.Sequential(
        [
            layer.InputLayer(input_shape=(latent_dim)),
            layer.Dense(units=8*512, activation='relu'),
            layer.Reshape((8, 512)),
            layer.Conv1DTranspose(filters=256, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Conv1DTranspose(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Conv1DTranspose(filters=64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layer.Conv1DTranspose(filters=6, kernel_size=3, strides=2, padding='same'),
        ], name="decoder1"
    )

    self.decoder2 = tf.keras.models.Sequential(
        [
            layer.InputLayer(input_shape=(latent_dim)),
            layer.Dense(units=8*512, activation='relu'),
            layer.Dense(512),
            layer.Dense(7, activation='softmax'),
        ], name="decoder2"
    )
  @property
  def metrics(self):
      return [
          self.total_loss_tracker,
          self.reconstruction_loss_tracker,
          self.kl_loss_tracker,
      ]

  def encode(self, x):
    return self.encoder(x)

  def decode(self, z):
    logits = self.decoder1(z)
    preds = self.decoder2(z)
    return logits, preds
  
  def call(self, x):
    codings = self.encode(x)
    return self.decoder1(codings)

  def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
        codings = self.encode(x)
        logits, preds = self.decode(codings)
        pred_loss = 0#tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y, preds), axis=1)
        #pred_loss = tf.reduce_sum(pred_loss, axis=1)
        pred_loss = tf.reduce_mean(pred_loss)
        r_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(x, logits), axis=1)
        r_loss = tf.reduce_mean(r_loss)
        loss = r_loss + pred_loss
        
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.total_loss_tracker.update_state(loss)
    self.reconstruction_loss_tracker.update_state(r_loss)
    self.kl_loss_tracker.update_state(pred_loss)
    return {
        "loss": self.total_loss_tracker.result(),
        "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        "pred_loss": self.kl_loss_tracker.result(),
    }

def ConvolutionalClassificationNN(num_labels):
    
    return tf.keras.models.Sequential(
        [
          #layer.Dropout(0.3),
          layer.Conv1D(filters=16, kernel_size=9, strides=1, padding='same'), #best model kernel size 9
          layer.Activation('relu'),
          layer.BatchNormalization(),
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
         
          layer.Conv1D(filters=32, kernel_size=7, strides=1, padding='same'), #best model kernel size 7
          layer.Activation('relu'),
          layer.BatchNormalization(),
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
          
          layer.Conv1D(filters=64, kernel_size=5, strides=1, padding='same'), #best model kernel size 5
          layer.Activation('relu'),
          layer.BatchNormalization(),
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
          
          layer.Conv1D(filters=128, kernel_size=3, strides=1, padding='same'), #best model kernel size 3
          layer.Activation('relu'),
          layer.BatchNormalization(),
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
         
          layer.Conv1D(filters=256, kernel_size=3, strides=1, padding='same'), #best model kernel size 3
          layer.Activation('relu'),
          layer.BatchNormalization(),
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
         
          layer.Flatten(),
          layer.Dropout(0.4),
          layer.Dense(num_labels, activation='softmax'),
        ]
    )

def ConvolutionalClassificationNNWithRNN(num_labels):
    
    return tf.keras.models.Sequential(
        [
          layer.Conv1D(filters=256, kernel_size=9, strides=1, activation='relu', padding='same'), #best model kernel size 9
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
         
          layer.Conv1D(filters=128, kernel_size=7, strides=1, activation='relu', padding='same'), #best model kernel size 7
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
          layer.GRU(40, return_sequences=True),
          layer.Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='same'), #best model kernel size 5
          layer.MaxPooling1D(pool_size=2, strides=2, padding='same'),
          layer.GRU(20, return_sequences=True),
         
          layer.Flatten(),
          layer.Dropout(0.4),
          layer.Dense(150),
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