import tensorflow as tf
from keras import layers

#convolutional layers
def convolutional_layer(input, num_filters, kernel_size=(3, 3), initializer='he_normal', dropout = False):

    #first convolution
    x = layers.Conv2D(filters=num_filters, kernel_size = kernel_size, padding='same', kernel_initializer=initializer)(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout:
        x = layers.Dropout(0.2)(x)

    #second convolution
    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x

#attention gate
def attention_gate(x, g, shape):
    #match dimensions of g and x
    g_down = layers.Conv2D(filters=shape, kernel_size=(1, 1), padding='same')(g)
    g_down = layers.BatchNormalization()(g_down)
    x_down = layers.Conv2D(filters=shape, kernel_size=(1, 1), padding='same')(x)
    x_down = layers.BatchNormalization()(x_down)

    #sum the features
    addition = layers.Add()([g_down, x_down])
    addition = layers.Activation('relu')(addition)

    #collapse the dimensions
    attention_map = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(addition)
    attention_map = layers.Activation('sigmoid')(attention_map)

    #final multiplication
    output = layers.Multiply()([x, attention_map])

    return output

#Channel Transformer class
class ChannelTransformer(layers.Layer):
    def __init__(self, num_heads=4, embed_dim=256, mlp_dim=256, dropout=0.5):
        super().__init__()

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dense(embed_dim)
        ])
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        H, W, C = x.shape[1], x.shape[2], x.shape[3]

        x = tf.reshape(x, (batch_size, -1, C))
        attn_output = self.attention(x, x)
        x = self.norm1(x + attn_output)

        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)

        x = tf.reshape(x, (batch_size, H, W, C))
        return x

#CCA class
class CCA(layers.Layer):
    def __init__(self, F_g, F_x):
        super(CCA, self).__init__()
        self.mlp_x = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(F_x, activation='relu')
        ])
        self.mlp_g = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(F_g, activation='relu')
        ])

    def call(self, g, x):
        channel_att_x = self.mlp_x(x)
        channel_att_g = self.mlp_g(g)

        scale = tf.expand_dims(tf.expand_dims(tf.nn.sigmoid((channel_att_x + channel_att_g) / 2.0), axis=1), axis=1)
        return x * scale