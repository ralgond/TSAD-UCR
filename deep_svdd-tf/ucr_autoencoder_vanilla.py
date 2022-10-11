from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPool1D,Input, UpSampling1D, Dense, Flatten, Reshape

def Vanilla_AE_pretrain(x_train, seq_len: int):
    assert (seq_len % 4 == 0)

    input = Input(shape=(seq_len, ))
    hidden = Dense(seq_len / 4, activation='relu', use_bias=False)(input)
    result = Dense(seq_len, activation='sigmoid', use_bias=False)(hidden)

    autoencoder = Model(inputs=input, outputs=result)
    encoder = Model(inputs=input, outputs=hidden)

    #autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(x_train, x_train, batch_size=64, epochs=5, shuffle=False)

    return encoder, autoencoder, history