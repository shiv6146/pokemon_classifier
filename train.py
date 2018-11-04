from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from scipy import misc
import numpy as np
import imageio
import json
import os

le = LabelEncoder()
batch_size = 30
epochs = 200

def get_class_names(prediction):
    return le.inverse_transform(prediction)

def load_data(data_dir='data'):
    train_x = []
    train_y = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            tmp_row = misc.imresize(imageio.imread(os.path.join(root, f), pilmode="L"), (32,32)).flatten()
            tmp_label = root.split('/')[1]

            train_x.append(tmp_row)
            train_y.append(tmp_label)
    return (np.array(train_x), np.array(train_y))

def train_and_save(x_train, y_train):
    x_train = x_train.reshape(600, 32 * 32)

    num_classes = np.unique(y_train).shape[0]

    # Encode labels
    y_train = le.fit_transform(y_train)
    y_train = to_categorical(y_train, num_classes)

    x_train = x_train.astype('float32')
    x_train /= 255

    # Train test split
    x_test = x_train[:100]
    y_test = y_train[:100]
    x_train = x_train[100:]
    y_train = y_train[100:]

    # Build model
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(32*32,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    # Train model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    print('Train loss:', history.history['loss'][-1])
    print('Train accuracy:', history.history['acc'][-1])

    # Save keras model for later prediction
    model.save('pokemon_classifier.h5')

    # Save label mappings
    label_dict = {a[0]: b for a, b in np.ndenumerate(le.inverse_transform([i for i in range(num_classes)]))}
    with open("mappings.json", "w") as fp:
        json.dump(label_dict, fp)
