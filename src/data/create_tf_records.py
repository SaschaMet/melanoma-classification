# Source: https://www.kaggle.com/animeshsinha1309/image-enhancement-and-tf-record-generation

import os
import cv2 as cv
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from matplotlib import pyplot as plt

PATH_TO_TF_RECORDS = os.getcwd() + '/tfrecords'


def hair_remove(image):
    # convert image to grayScale
    grayScale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # kernel for morphologyEx
    kernel = cv.getStructuringElement(1, (17, 17))
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv.morphologyEx(grayScale, cv.MORPH_BLACKHAT, kernel)
    # apply thresholding to blackhat
    _, threshold = cv.threshold(blackhat, 10, 255, cv.THRESH_BINARY)
    # inpaint with original image and threshold image
    final_image = cv.inpaint(image, threshold, 1, cv.INPAINT_TELEA)

    return final_image


def create_tfrecords_dir():
    if not os.path.exists(PATH_TO_TF_RECORDS):
        os.mkdir(PATH_TO_TF_RECORDS)


def prepare_df(df, is_train):

    # Mean Encode the Age NaN values
    df.age_approx.fillna(df.age_approx.mean(), inplace=True)
    df['age_approx'] = df.age_approx.astype('int')

    # Label Encode all the strings
    labels_categorical = ['patient_id', 'sex',
                          'anatom_site_general_challenge', 'age_approx']
    for label in labels_categorical:
        df[label], _ = df[label].factorize()

    if is_train:
        df.diagnosis, _ = df.diagnosis.factorize()
        df = df.drop(['benign_malignant'], axis=1)

    return df


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_tf_train(base_path, df, writer):
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_path = base_path + "/train/" + row['image_name'] + ".jpg"

        img = cv.imread(img_path)
        img = cv.resize(img, (1024, 1024), interpolation=cv.INTER_AREA)
        img = hair_remove(img)
        img = cv.imencode('.jpg', img, (cv.IMWRITE_JPEG_QUALITY, 94))[
            1].tostring()

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image': bytes_feature(img),
            'image_name': bytes_feature(str.encode(row['image_name'])),
            'patient_id': int64_feature(row['patient_id']),
            'sex': int64_feature(row['sex']),
            'age_approx': int64_feature(row['age_approx']),
            'anatom_site_general_challenge': int64_feature(row['anatom_site_general_challenge']),
            'diagnosis': int64_feature(row['diagnosis']),
            'target': int64_feature(row['target'])
        }))
        writer.write(tf_example.SerializeToString())


def create_tf_test(base_path, df, writer):
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_path = base_path + "/train/" + row['image_name'] + ".jpg"

        img = cv.imread(img_path)
        img = cv.resize(img, (1024, 1024), interpolation=cv.INTER_AREA)
        img = hair_remove(img)
        img = cv.imencode('.jpg', img, (cv.IMWRITE_JPEG_QUALITY, 94))[
            1].tostring()

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image': bytes_feature(img),
            'image_name': bytes_feature(str.encode(row['image_name'])),
            'patient_id': int64_feature(row['patient_id']),
            'sex': int64_feature(row['sex']),
            'age_approx': int64_feature(row['age_approx']),
            'anatom_site_general_challenge': int64_feature(row['anatom_site_general_challenge']),
        }))
        writer.write(tf_example.SerializeToString())


def create_tf_record_from_df(base_path, input_df, output_path):
    create_tfrecords_dir()
    writer = tf.io.TFRecordWriter(output_path)

    is_train = 'train' in output_path

    df = prepare_df(input_df, is_train)

    if is_train:
        create_tf_train(base_path, df, writer)
    else:
        create_tf_test(base_path, df, writer)

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))
