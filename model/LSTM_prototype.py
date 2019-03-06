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

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]
file_path = 'saved_models/lstm_weights.h5'
callbacks = get_callbacks(filepath=file_path, patience=5)

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
    global SEQ_LEN, SEQ_STRIDE

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
    y = keras.utils.to_categorical(y, num_classes=6)

    return {'x': x, 'y': y}

def build_model():
    model = Sequential()
    model.add(LSTM(32, input_shape=(1,SEQ_LEN)))
    model.add(Dense(6))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def main():
    gps_files = glob.glob('../data/prototype/**/gps_points.csv')
    trip_files = glob.glob('../data/prototype/**/gps_trips.csv')

    file_results = process_file(trip_file = trip_files[0], gps_file = gps_files[0])
    data = build_seq(input_df = file_results['df'], unique_trips = file_results['unique_trips'])

    x_train, x_val, y_train, y_val = train_test_split(data['x'], data['y'], random_state=1, train_size=0.8)

    model = build_model()

    model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

if __name__ == '__main__':
    main()
