import pandas as pd
import numpy as np
import glob

from sklearn.model_selection import train_test_split

from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

conf = SparkConf().setAppName('Random_Forrest_Spark').setMaster('spark://192.168.29.108:7077')
sc = SparkContext(conf=conf)

spark = SparkSession.builder.getOrCreate()

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
#    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
#    y = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

    return {'x': x, 'y': y}


def main():
    gps_files = glob.glob('/home/vmuser/Downloads/caltrans_sorted_by_person/**/gps_points.csv')
    trip_files = glob.glob('/home/vmuser/Downloads/caltrans_sorted_by_person/**/gps_trips.csv')
#    gps_files = glob.glob('../data/prototype/**/gps_points.csv')
#    trip_files = glob.glob('../data/prototype/**/gps_trips.csv')

    file_results = process_file(trip_file = trip_files[0], gps_file = gps_files[0])
    seq_results = build_seq(input_df = file_results['df'], unique_trips = file_results['unique_trips'])

    X = seq_results['x']
    y = seq_results['y']

    print('Bulding training data from files..')
    for i in range(1, len(gps_files[0:200])):
        if(i%20 == 0):
            print('loaded ', i, 'files out of ', len(gps_files[0:200]))
        file_results = process_file(trip_file = trip_files[i], gps_file = gps_files[i])
        seq_results = build_seq(input_df = file_results['df'], unique_trips = file_results['unique_trips'])

        X = np.vstack((X, seq_results['x']))
        y = np.vstack((y, seq_results['y']))

    x_train, x_val, y_train, y_val = train_test_split(X, y, random_state=1, train_size=0.8)


    train_inputs = np.hstack((y_train, x_train))
    train_vectors = map(lambda x: (int(x[0]), Vectors.dense(x[1:])), train_inputs)
    train_df = spark.createDataFrame(train_vectors,schema=["label", "features"])

    test_vectors = map(lambda x: (Vectors.dense(x),), x_val)
    test_df = spark.createDataFrame(test_vectors,schema=["features"])
#    test_df.show(5)

    rf = RandomForestClassifier(labelCol='label', featuresCol='features')

    model = rf.fit(train_df)

    predictions = model.transform(test_df)
#    predictions.show(20)

    y_pred = np.array(predictions.select("prediction").rdd.flatMap(lambda x: x).collect())
    y_pred = y_pred.reshape(-1, 1)

    print("Validation Accuracy: ", sum(y_pred == y_val)/y_pred.shape[0])

if __name__ == '__main__':
    main()
