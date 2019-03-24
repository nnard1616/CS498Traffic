import pandas as pd
import numpy as np
import glob

import keras
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

#Global configuration variables
SEQ_LEN = 50
SEQ_STRIDE = 1
NUM_CLASSES = 30

def process_file(trip_file, gps_file):
    trip_df = pd.read_csv(trip_file)
    gps_df = pd.read_csv(gps_file)

    trip_df = trip_df[['sampno', 'perno', 'gpstripid', 'travel_mode']]
    gps_df = gps_df[['sampno', 'perno', 'gpstripid', 'time_local', 'gpsspeed']]

    unique_trips = list()
    for i, row in trip_df.iterrows():
        unique_trips.append("sampno == '{}' and perno == '{}' and gpstripid == '{}'".format(row.sampno, row.perno, row.gpstripid))

    merged_df = pd.merge(trip_df, gps_df,  how='left', on=['sampno', 'perno', 'gpstripid'])
    merged_df = merged_df[['sampno', 'perno', 'gpstripid', 'time_local', 'gpsspeed', 'travel_mode']]

    return {'df': merged_df, 'unique_trips': unique_trips}

def build_seq(input_df, unique_trips):
    global SEQ_LEN, SEQ_STRIDE, NUM_CLASSES

    x = np.ndarray([1,SEQ_LEN])
    y = np.ndarray([1,1], dtype=int)

    for trip in unique_trips:
        df = input_df.query(trip)
        if len(df.index) < SEQ_LEN:
            continue
        x_tmp = np.zeros([len(df.index)//SEQ_STRIDE-SEQ_LEN,SEQ_LEN])
        y_tmp = np.zeros([len(df.index)//SEQ_STRIDE-SEQ_LEN,1])
        for i in range(0, len(df.index)-SEQ_LEN, SEQ_STRIDE):
            x_tmp[i] = df.gpsspeed.iloc[i:i+SEQ_LEN]
            y_tmp[i] = df.travel_mode.iloc[i+SEQ_LEN]
        x = np.append(x, x_tmp, axis=0)
        y = np.append(y, y_tmp, axis=0)

    x = np.delete(x, 0, axis=0)
    y = np.delete(y, 0, axis=0)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    y = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

    return {'x': x, 'y': y}

def build_model(layer_units=[128], 
                dropout=True,
                dropout_level=0.5,
                gpu=False,
                seq_len=SEQ_LEN,
                num_classes=NUM_CLASSES):

    assert len(layer_units) > 0, 'Must include at least 1 layer'

    model = Sequential()
    for i, units in enumerate(layer_units):
        if gpu:
            layer = CuDNNLSTM(units, input_shape=(1,seq_len))
        else:
            layer = LSTM(units, input_shape=(1,seq_len))
        model.add(layer)
        if dropout and (i != len(layer_units)-1):
            model.add(Dropout(dropout_level))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def main():
    gps_files = glob.glob('../data/prototype/**/gps_points.csv')
    trip_files = glob.glob('../data/prototype/**/gps_trips.csv')

    file_results = process_file(trip_file = trip_files[0], gps_file = gps_files[0])
    seq_results = build_seq(input_df = file_results['df'], unique_trips = file_results['unique_trips'])

    X = seq_results['x']
    y = seq_results['y']

    print('Bulding training data from files..')
    for i in range(1, len(gps_files)):
        file_results = process_file(trip_file = trip_files[i], gps_file = gps_files[i])
        seq_results = build_seq(input_df = file_results['df'], unique_trips = file_results['unique_trips'])

        X = np.vstack((X, seq_results['x']))
        y = np.vstack((y, seq_results['y']))

    x_train, x_val, y_train, y_val = train_test_split(X, y, random_state=1, train_size=0.8)

    model = build_model()

    model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

    y_pred = model.predict(x_val)

    acc = sum(np.argmax(y_pred, axis=1) == np.argmax(y_val, axis=1)) / y_pred.shape[0]

    print("Validation Accuracy: {number:.{digits}f}%".format(number=(acc*100), digits=2))


if __name__ == '__main__':
    main()
