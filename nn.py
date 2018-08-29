from keras.models import Model
from keras.layers import LSTM
from keras.layers import Embedding, Dense, TimeDistributed, Lambda
from keras.layers import Input, RepeatVector, Dropout, Concatenate
from keras.applications.mobilenet import MobileNet
from keras.optimizers import RMSprop

from stuff import E_DIM, Z_DIM, RESIZE, PERMUTE, BATCH_SIZE, BATCHES, EPOCHS, MAXLEN, PAD_EMPTY, SHIFT
from stuff import get_flickr8k
from flickr8k import shifted_cropped_batch, batch_generator

f8k = get_flickr8k()

# caption encoding
caption_input = Input([None], name='caption_input')

e = Embedding(f8k.voclen, E_DIM)(caption_input)

caption_encoder = Model(caption_input, e, name='caption_encoder')

# image encoding
image_input = Input(list(RESIZE) + [3], name='image_input')

mobilenet = MobileNet(
        list(RESIZE) + [3],
        include_top=False,
        weights='imagenet',
        pooling='avg'
)

for layer in mobilenet.layers:
    layer.trainable = False

z = mobilenet(image_input)
z = Dense(Z_DIM, name='image_z')(z)

image_encoder = Model(image_input, z, name='image_encoder')

# caption decoder

m = Concatenate(1)([
        RepeatVector(1)(image_encoder(image_input)),
        caption_encoder(caption_input)
])
hs = LSTM(
        100, return_sequences=True
)(m)
hs = Lambda(lambda x: x[:, 1:, :])(hs)
hyp_c = TimeDistributed(Dense(f8k.voclen, activation='softmax'))(hs)

# model instantiation
model = Model(
        [image_input, caption_input],
        hyp_c
)

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

# batch generator
next_train_batch = batch_generator(
        shifted_cropped_batch, f8k=f8k, batch_size=BATCH_SIZE,
        maxlen=MAXLEN, pad_empty=PAD_EMPTY, shift=SHIFT,
        idxs=f8k.train_idx, permute=PERMUTE)

validation_batch = next(batch_generator(
        shifted_cropped_batch, f8k=f8k, batch_size=1000,
        maxlen=MAXLEN, pad_empty=PAD_EMPTY, shift=SHIFT,
        idxs=f8k.test_idx, permute=PERMUTE)())


model.fit_generator(next_train_batch(), validation_data=validation_batch,
            steps_per_epoch=BATCHES, epochs=EPOCHS, verbose=1
)

model.save('ic1.h5')


