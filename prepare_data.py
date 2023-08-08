import tensorflow as tf
import numpy as np
import IPython.display as display

from preprocessing import load_audio_array, split_equal_size, load_spec_array

train_o = load_spec_array('../spec_train_o')
#train_o_split = split_equal_size(train_o)  # hier werden die daten in gleiche laenge aufgesteilt

#train_r = load_spec_array('../spec_train_r')
train_r_split = split_equal_size(train_r)

#val_o = load_spec_array('../spec_val_o')
#val_o_split = split_equal_size(val_o)

#val_r = load_spec_array('../spec_val_r')
#val_r_split = split_equal_size(val_r)


#tf.io.serialize_tensor()
#tf.io.parse_tensor()

# feature0: id, feature1: name, feature2: original, feature3: remix
def serialize_example(id, name, original, remix):
    feature = {
        'id': id,
        'name': name,
        'original': original,
        'remix': remix,
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

for tensor in train_o_split:
    serial_tensor = tf.io.serialize_tensor(tensor)
    feature_of_tensor = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serial_tensor.numpy()]))