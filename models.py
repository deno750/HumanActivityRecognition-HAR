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
def Decoder(output_dim):
  return tf.keras.models.Sequential([
    layer.Dense(8 * 256, activation='relu'),
    layer.Reshape((8, 256)),
    layer.Conv1DTranspose(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'),
    layer.Conv1DTranspose(filters=64, kernel_size=5, strides=2, activation='relu', padding='same'),
    layer.Conv1DTranspose(filters=32, kernel_size=7, strides=2, activation='relu', padding='same'),
    #layer.Conv1DTranspose(filters=16, kernel_size=9, strides=2, activation='relu', padding='same'),
    layer.Conv1DTranspose(filters=output_dim, kernel_size=3, strides=2, padding='same'),
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

  def test_step(self, data):
    x, _ = data
    mean, logvar = self.encode(x)
    z = self.reparameterize(mean, logvar)
    logits = self.decode(z)
    kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
    kl_loss = tf.reduce_sum(kl_loss, axis=1)
    kl_loss = tf.reduce_mean(kl_loss)
    r_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(x, logits), axis=1)
    r_loss = tf.reduce_mean(r_loss)
    loss = r_loss + kl_loss  
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

class InceptionBlockV2(tf.keras.layers.Layer):
  def __init__(self, num_filters, activation='relu'):
      super().__init__()
      self.activation = layer.Activation(activation)
      self.num_filters = num_filters
      #Branch 1
      self.bottleneck = layer.Conv1D(filters=num_filters, kernel_size=1, activation='relu', padding='same')
      self.conv1 = layer.Conv1D(filters=num_filters, kernel_size=10, activation='relu', padding='same')
      self.conv2 = layer.Conv1D(filters=num_filters, kernel_size=20, activation='relu', padding='same')
      self.conv3 = layer.Conv1D(filters=num_filters, kernel_size=40, activation='relu', padding='same')
      #Branch 2
      self.pool = layer.MaxPooling1D(pool_size=3, strides=1, padding='same')
      self.conv4 = layer.Conv1D(filters=num_filters, kernel_size=1, strides=1, activation='relu', padding='same')

      self.bn = layer.BatchNormalization()

  def call(self, x):
    branch1 = self.bottleneck(x)
    out_conv1 = self.conv1(branch1)
    out_conv2 = self.conv2(branch1)
    out_conv3 = self.conv3(branch1)

    branch2 = self.pool(x)
    out_conv4 = self.conv4(branch2)
    Z = tf.concat(values=[out_conv1, out_conv2, out_conv3, out_conv4], axis=-1)
    Z = self.bn(Z)
    return self.activation(Z)

  def get_config(self):
    config = super(InceptionBlockV2, self).get_config()
    config.update({"num_filters": self.num_filters})
    return config

class InceptionBlock(tf.keras.layers.Layer):
  def __init__(self):
      super().__init__()
      #Branch 1
      self.b1_1x1 = layer.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same')
      #Branch 2
      self.b2_1x1 = layer.Conv1D(filters=32, kernel_size=1, activation='relu', padding='same')
      self.b2_1x3 = layer.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
      #Branch 3
      self.b3_1x1 = layer.Conv1D(filters=32, kernel_size=1, activation='relu', padding='same')
      self.b3_1x5 = layer.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')
      #Branch 4
      self.b4_pool = layer.MaxPooling1D(pool_size=3, strides=1, padding='same')
      self.b4_1x1 = layer.Conv1D(filters=64, kernel_size=1, strides=1, activation='relu', padding='same')

  def call(self, x):
    branch1 = self.b1_1x1(x)

    branch2 = self.b2_1x1(x)
    branch2 = self.b2_1x3(branch2)

    branch3 = self.b3_1x1(x)
    branch3 = self.b3_1x5(branch3)

    branch4 = self.b4_pool(x)
    branch4 = self.b4_1x1(x)

    return tf.concat(values=[branch1, branch2, branch3, branch4], axis=-1)

def InceptionModel(num_labels):
    return tf.keras.Sequential([
        InceptionBlock(),
        InceptionBlock(),
        InceptionBlock(),
        layer.MaxPooling1D(pool_size=2, padding='same'),
        InceptionBlock(),
        layer.MaxPooling1D(pool_size=2, padding='same'),
        layer.GRU(120, return_sequences=True),
        layer.GRU(40),
        layer.Dense(num_labels, activation='softmax'),
    ])

def skip_layer(input, Z_inception):
    Z_short = layer.Conv1D(filters=Z_inception.shape[-1], kernel_size=1, padding='same', use_bias=False)(input)
    Z_short = layer.BatchNormalization()(Z_short)
    Z_out = layer.Add()([Z_inception, Z_short])
    return layer.Activation('relu')(Z_out)

def buildInceptionModelV2(input_shape, num_classes, num_modules):
    input_layer = tf.keras.layers.Input(input_shape)
    Z = input_layer
    Z_res = input_layer
    for i in range(num_modules):
        Z = InceptionBlockV2(num_filters=32)(Z)
        if i % 3 == 0:
            Z = skip_layer(Z_res, Z)
            Z = Z_res
    gap_layer = layer.GlobalAveragePooling1D()(Z)
    output_layer = layer.Dense(num_classes, activation='softmax')(gap_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


class SSAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, encoder, decoder, classifier, alpha=0.1, beta=1.0, **kwargs):
    super(SSAE, self).__init__(**kwargs)
    self.latent_dim = latent_dim
    self.alpha = alpha
    self.beta = beta
    self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
    self.kl_loss_tracker = tf.keras.metrics.Mean(name="pred_loss")
    self.encoder = encoder
    self.decoder = decoder
    self.classifier = classifier
    
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
    logits = self.decoder(z)
    preds = self.classifier(z)
    return logits, preds

  def predict(self, x):
      codings = self.encode(x)
      return self.classifier(codings)
  
  def call(self, x):
    codings = self.encode(x)
    return self.decoder(codings)

  def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
        codings = self.encode(x)
        logits, preds = self.decode(codings)
        pred_loss = tf.keras.losses.sparse_categorical_crossentropy(y, preds)
        pred_loss = tf.reduce_mean(pred_loss)
        r_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(x, logits), axis=1)
        r_loss = tf.reduce_mean(r_loss)
        loss = self.alpha * r_loss + self.beta * pred_loss
        
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.total_loss_tracker.update_state(loss)
    self.reconstruction_loss_tracker.update_state(r_loss)
    self.kl_loss_tracker.update_state(pred_loss)
    return {
        "loss": self.total_loss_tracker.result(),
        "rec_loss": self.reconstruction_loss_tracker.result(),
        "pred_loss": self.kl_loss_tracker.result(),
    }
    
  def test_step(self, data):
    x, y = data
    codings = self.encode(x)
    logits, preds = self.decode(codings)
    pred_loss = tf.keras.losses.sparse_categorical_crossentropy(y, preds)
    pred_loss = tf.reduce_mean(pred_loss)
    r_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(x, logits), axis=1)
    r_loss = tf.reduce_mean(r_loss)
    loss = self.alpha * r_loss + self.beta * pred_loss    
    self.total_loss_tracker.update_state(loss)
    self.reconstruction_loss_tracker.update_state(r_loss)
    self.kl_loss_tracker.update_state(pred_loss)
    return {
        "loss": self.total_loss_tracker.result(),
        "rec_loss": self.reconstruction_loss_tracker.result(),
        "pred_loss": self.kl_loss_tracker.result(),
    }
  
  def save_model(self, name):
    self.encoder.save("models/" +name+ "_encoder" + ".h5")
    self.decoder.save("models/" +name+ "_decoder" + ".h5")
    self.classifier.save("models/" +name+ "_classifier" + ".h5")