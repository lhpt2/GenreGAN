import tensorflow as tf
import numpy as np
import IPython.display as display

from constants import secs_to_bins, bins_to_secs
from preprocessing import split_equal_size, load_spec_o_r_arrays, load_spec_array, enlarge_to_longest_spec, shorten_to_shortest_spec
from testing_network import save_spec_to_wv


#train_o = load_spec_array('../spec_train_o')
#train_o_split = split_equal_size(train_o)  # hier werden die daten in gleiche laenge aufgesteilt

#train_r = load_spec_array('../spec_train_r')
#train_r_split = split_equal_size(train_r)

#val_o = load_spec_array('../spec_val_o')
#val_o_split = split_equal_size(val_o)

#val_r = load_spec_array('../spec_val_r')
#val_r_split = split_equal_size(val_r)


#tf.io.serialize_tensor()
#tf.io.parse_tensor()

# feature0: id, feature1: name, feature2: original, feature3: remix

def _tensor_feature(tensor):
    serial_tensor = tf.io.serialize_tensor(tensor)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serial_tensor]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _string_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(id: int, name: str, original: np.ndarray, remix: np.ndarray):
    feature = {
        'id': _int64_feature(id),
        'name': _string_feature(name),
        'original': _tensor_feature(original),
        'remix': _tensor_feature(remix),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(id: int, name: str, original: np.ndarray, remix: np.ndarray):
  tf_string = tf.py_function(
    serialize_example,
    (id, name, original, remix),  # Pass these args to the above function.
    tf.string)      # The return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar.

def _parse_function(example_proto):
    # Create a description of the features.
    feature_description = {
        'id': tf.io.FixedLenFeature([], tf.int32, default_value=0),
        'name': tf.io.FixedLenFeature([], tf.string, default_value=0),
        'original': tf.io.FixedLenFeature([], tf.float32, default_value=''),
        'remix': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

if __name__ == '__main__':
    filename = 'spectrals_train.tfrecord'
    print(secs_to_bins(5))
    print(secs_to_bins(1))
    print(bins_to_secs(72))
    exit()
    ids, names, spec_train_o, spec_train_r = load_spec_o_r_arrays('../spec_train_o', '../spec_train_r')

    ###### PROBLEM: DIE DATEN BZW SONGS MUESSEN GLEICH LANG SEIN, DAMIT SIE IN EINEN TENSOR PASSEN

    #spec_train_o = split_equal_size(spec_train_o)
    #spec_train_r = split_equal_size(spec_train_r)
    #spec_train_o = enlarge_to_longest_spec(spec_train_o)
    #spec_train_r = enlarge_to_longest_spec(spec_train_r)
    spec_train_o = shorten_to_shortest_spec(spec_train_o)
    spec_train_r = shorten_to_shortest_spec(spec_train_r)

    ds = tf.data.Dataset.from_tensor_slices((ids, names, spec_train_o, spec_train_r))
    for f0, f1, f2, f3 in ds.take(1):
        print(int(f0))
        print(f1.numpy().decode())
        print(f2.shape)
        print(f3.shape)

    #s = tf.io.serialize_tensor(spec_train_o[0])
    #print(type(s))


    #id_ds = tf.data.Dataset.from_tensor_slices(ids)
    #names_ds = tf.data.Dataset.from_tensor_slices(names)
    #ft_ds = tf.data.Dataset.from_tensor_slices((ids, names, spec_train_o, spec_train_r))
    #serialized_ds = ft_ds.map(tf_serialize_example)
    #writer = tf.data.experimental.TFRecordWriter(filename)
    #writer = tf.io.TFRecordWriter(filename)
    #writer.write(serialized_ds)
    #dataset = tf.data.TFRecordDataset(filename) # . all ops to be done here
    #raw_example = next(iter(dataset))
    #parsed = tf.train.Example.FromString(raw_example.numpy())
    #print(parsed.features.feature['name'].bytes_list.value)
    #rr = tf.io.parse_tensor(parsed.features.feature['remix'].bytes_list.value, tf.float32)
    #remix = tf.io.parse_tensor(rr, tf.float32)
    #save_spec_to_wv(remix, './remix.wav')