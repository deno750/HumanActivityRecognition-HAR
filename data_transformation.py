import tensorflow as tf
import numpy as np

def frame_data(input_data, input_labels, frame_len=128, frame_step=8):
    data_len = input_data.shape[0]
    X_tensor = []
    y_tensor = []
    for i in range(0, data_len):
        data = input_data[i].T
        #print("Data shape")
        #print(data.shape)
        frames = tf.signal.frame(data, frame_length=frame_len, frame_step=frame_step, pad_end=True)
        #print("Frame shape")
        #print(frames.shape)
        frames = tf.cast(frames, dtype=tf.float32)
        frames = tf.transpose(frames, [1, 2, 0]) #tf.transpose(frames, [1, 0, 2])
        tot_measures = frames.shape[0]
        for j in range(0, tot_measures):
            X_tensor.append(frames[j])
            y_tensor.append(input_labels[i])
    return (np.array(X_tensor), np.array(y_tensor))