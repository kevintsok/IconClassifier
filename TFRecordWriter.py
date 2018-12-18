from random import shuffle
import glob
import tensorflow as tf
import sys
import cv2
import numpy as np

shuffle_data = True  # shuffle the addresses before saving
train_ratio = 0.9
val_ratio   = 0.05
test_ratio  = 0.05

images_path = './single/output/*.jpg'
# read addresses and labels from the data folder
addrs = glob.glob(images_path)
labels = [int(addr.split('/')[-1].split('.')[0][16:])-1 for addr in addrs]

# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

l = len(labels)
print(len(labels),len(addrs))
print(labels.count(labels==0))

# Divide the data
train_addrs = addrs[0:int(train_ratio * l)]
train_labels = labels[0:int(train_ratio * l)]
val_addrs = addrs[int(train_ratio * l):int((train_ratio+val_ratio) * l)]
val_labels = labels[int(train_ratio * l):int((train_ratio+val_ratio) * l)]
test_addrs = addrs[int((train_ratio+val_ratio) * l):]
test_labels = labels[int((train_ratio+val_ratio) * l):]

print(test_labels[1:10])
print(test_addrs[1:10])

def load_image(addr):
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return tf.compat.as_bytes(img.tostring())

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def save_to_tfrecords(addrs,labels,mode):


    filename = mode+'.tfrecords'  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(addrs)):
    # print how many images are saved every 1000 images
        if not i % 1000:
            print((mode+' data: {}/{}').format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(addrs[i])
        label = labels[i]
        # Create a feature
        labelname = mode+'/label'
        imagename = mode + '/image'
        feature = {labelname: _int64_feature(label),
                   imagename: _bytes_feature(img)}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

#save_to_tfrecords(train_addrs, train_labels, 'train')
#save_to_tfrecords(test_addrs, test_labels, 'test')