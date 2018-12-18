from random import shuffle
import glob
import tensorflow as tf
import sys
#import cv2
import numpy as np



images_path = './single/output/*.jpg'
data_path = 'train.tfrecords'

def _parse_(serialized_example,mode):
    labelname = mode + '/label'
    imagename = mode + '/image'
    feature = {imagename: tf.FixedLenFeature([], tf.string),
               labelname: tf.FixedLenFeature([], tf.int64)}
    example = tf.parse_single_example(serialized_example,feature)
    image = tf.decode_raw(example[imagename],tf.float32) #remember to parse in int64. float will raise error
    label = tf.cast(example[labelname],tf.int32)
    return (dict({'image':image}),label)


def tfrecord_input_fn(batch_size, mode):
    data_path = mode+'.tfrecords'
    print(data_path)
    tfrecord_dataset = tf.data.TFRecordDataset(data_path)
    tfrecord_dataset = tfrecord_dataset.map(lambda x: _parse_(x,mode))
    #tfrecord_dataset = tfrecord_dataset.shuffle(True)
    tfrecord_dataset = tfrecord_dataset.batch(batch_size)
    tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
    return tfrecord_iterator.get_next()

feature_column = [tf.feature_column.numeric_column(key='image',shape=(30000,))]  #100*100*3


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["image"], [-1, 100, 100, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # max Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool3_flat = tf.contrib.layers.flatten(pool3)
    dense = tf.layers.dense(inputs=pool3_flat, units=2048, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dense, units=64)  #one_hot

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits, name="softmax_tensor"),
        'logits': logits,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=64)
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    tf.summary.scalar('accuracy', accuracy[1])
    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {"accuracy": accuracy}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)


#model=tf.estimator.DNNClassifier([100,100,3],n_classes=64,feature_columns=feature_column)
#model.train(l,steps=2000)
classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="cnn_model")

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


classifier.train(input_fn=lambda: tfrecord_input_fn(16,'train'), steps=50000, hooks=[logging_hook])
evalution = classifier.evaluate(input_fn=lambda: tfrecord_input_fn(1,'test'))

print(evalution)

