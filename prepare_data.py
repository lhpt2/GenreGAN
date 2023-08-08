import tensorflow as tf
import numpy as np
import IPython.display as display

from preprocessing import split_equal_size, load_spec_o_r_arrays

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
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serial_tensor.numpy()]))

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
        'name': _string_feature(name.encode()),
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


if __name__ == '__main__':
    filename = 'spectrals_train.tfrecord'
    ft_ds = tf.data.Dataset.from_tensor_slices(load_spec_o_r_arrays('../spec_train_o', '../spec_train_r'))
    f0, f1, f2, f3 = ft_ds.take(1)
    print(f0)
    print(f1)
    print(f2)
    print(f3)
    serialized_ds = ft_ds.map(tf_serialize_example)
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_ds)
