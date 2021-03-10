# Write-TFRecords---Train
# Source: https: // www.kaggle.com/cdeotte/how-to-create-tfrecords
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example_train(
    image,
    image_name,
    patient_id,
    sex_female,
    sex_male,
    image_path,
    age_approx,
    anatom_head_neck,
    anatom_lower_extremity,
    anatom_oral_genital,
    anatom_palms_soles,
    anatom_torso,
    anatom_upper_extremity,
    target
):
    feature = {
        'image': _bytes_feature(image),
        'image_name': _bytes_feature(image_name),
        'patient_id': _bytes_feature(patient_id),
        'sex_female': _int64_feature(sex_female),
        'sex_male': _int64_feature(sex_male),
        'image_path': _bytes_feature(image_path),
        'age_approx': _int64_feature(age_approx),
        'anatom_head_neck': _int64_feature(anatom_head_neck),
        'anatom_lower_extremity': _int64_feature(anatom_lower_extremity),
        'anatom_oral_genital': _int64_feature(anatom_oral_genital),
        'anatom_palms_soles': _int64_feature(anatom_palms_soles),
        'anatom_torso': _int64_feature(anatom_torso),
        'anatom_upper_extremity': _int64_feature(anatom_upper_extremity),
        'target': _int64_feature(target)
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def serialize_example_test(
    image,
    image_name,
    patient_id,
    sex_female,
    sex_male,
    image_path,
    age_approx,
    anatom_head_neck,
    anatom_lower_extremity,
    anatom_oral_genital,
    anatom_palms_soles,
    anatom_torso,
    anatom_upper_extremity
):
    feature = {
        'image': _bytes_feature(image),
        'image_name': _bytes_feature(image_name),
        'patient_id': _bytes_feature(patient_id),
        'sex_female': _int64_feature(sex_female),
        'sex_male': _int64_feature(sex_male),
        'image_path': _bytes_feature(image_path),
        'age_approx': _int64_feature(age_approx),
        'anatom_head_neck': _int64_feature(anatom_head_neck),
        'anatom_lower_extremity': _int64_feature(anatom_lower_extremity),
        'anatom_oral_genital': _int64_feature(anatom_oral_genital),
        'anatom_palms_soles': _int64_feature(anatom_palms_soles),
        'anatom_torso': _int64_feature(anatom_torso),
        'anatom_upper_extremity': _int64_feature(anatom_upper_extremity)
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tf_records_train(df, size_of_record=200):
    number_of_images = df.shape[0]
    number_of_tf_record_files = number_of_images//size_of_record + \
        int(number_of_images % size_of_record != 0)

    for i in range(number_of_tf_record_files):
        print()
        print('Writing TFRecord', i + 1, 'of', number_of_tf_record_files)
        Images_In_TF_Records = min(
            size_of_record, number_of_images - i * size_of_record)
        print('Images_In_TF_Records', Images_In_TF_Records)
        with tf.io.TFRecordWriter('tfrecords/train%.2i-%i.tfrec' % (i, Images_In_TF_Records)) as writer:
            for k in range(Images_In_TF_Records):
                offset = size_of_record * i
                # get row image_path in df
                row = df.iloc[offset + k]
                image_path = row['image_path']  # get row image_path in df
                img = cv2.imread(image_path)

                # we need this option turned on when working with the original dataset
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # fix incorrect colors

                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[
                    1].tostring()
                example = serialize_example_train(
                    img,
                    str.encode(row['image_name']),
                    str.encode(row['patient_id']),
                    row['sex_female'],
                    row['sex_male'],
                    str.encode(row['image_path']),
                    row['age_approx'],
                    row['anatom_head_neck'],
                    row['anatom_lower_extremity'],
                    row['anatom_oral_genital'],
                    row['anatom_palms_soles'],
                    row['anatom_torso'],
                    row['anatom_upper_extremity'],
                    row['target']
                )
                writer.write(example)
                if k % 100 == 0:
                    print(k, ', ', end='')


def write_tf_records_test(df, size_of_record=200):
    number_of_images = df.shape[0]
    number_of_tf_record_files = number_of_images//size_of_record + \
        int(number_of_images % size_of_record != 0)

    for i in range():
        print()
        print('Writing TFRecord', i + 1, 'of', number_of_tf_record_files)
        Images_In_TF_Records = min(
            size_of_record, number_of_images - i * size_of_record)
        print('Images_In_TF_Records', Images_In_TF_Records)
        with tf.io.TFRecordWriter('./tfrecords/test%.2i-%i.tfrec' % (i, Images_In_TF_Records)) as writer:
            for k in range(Images_In_TF_Records):
                offset = size_of_record * i
                row = df.iloc[offset + k]  # get row image_path in df
                image_path = row['image_path']  # get row image_path in df
                img = cv2.imread(image_path)

                # we need this option turned on when working with the original dataset
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # fix incorrect colors

                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[
                    1].tostring()
                example = serialize_example_test(
                    img,
                    str.encode(row['image_name']),
                    str.encode(row['patient_id']),
                    row['sex_female'],
                    row['sex_male'],
                    str.encode(row['image_path']),
                    row['age_approx'],
                    row['anatom_head_neck'],
                    row['anatom_lower_extremity'],
                    row['anatom_oral_genital'],
                    row['anatom_palms_soles'],
                    row['anatom_torso'],
                    row['anatom_upper_extremity']
                )
                writer.write(example)
                if k % 100 == 0:
                    print(k, ', ', end='')
