import sys
import tensorflow as tf
import numpy as np
from src.load_data import *
from src.glove import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim

GLOVE_DIMENSION = 50
MAX_SEQ_LEN = 120


def prep_data(DATA_FILE, PERCENTAGE_TRAIN):
    def pad(sentence):
        if (len(sentence) > MAX_SEQ_LEN):
            return sentence[:MAX_SEQ_LEN]
        else:
            num_pads = MAX_SEQ_LEN - len(sentence)
            return sentence + [""] * num_pads

    def process_rows(data, glove_vectors):
        """
            Generate input x matrix M[num_data_points][glove_dimension][max_seq_length]
        """

        inputs, outputs = [], []
        for sentence, mbti_hot in data:
            x = []
            ### Pad sentence
            padded_sentence = pad(sentence)

            ### Convert each word in sentence to glove vector
            for w in padded_sentence:
                embedding = np.array([0.0] * GLOVE_DIMENSION)
                if (w in glove_vectors):
                    embedding = glove_vectors[w]
                x.append(embedding)

            inputs.append(x)
            outputs.append(mbti_hot)

        return inputs, outputs

    train_data, test_data = load_data(DATA_FILE, PERCENTAGE_TRAIN)
    glove_vectors = load_word_vectors("../data/glove.6B.50d.txt", GLOVE_DIMENSION)
    train_x, train_y = process_rows(train_data, glove_vectors)
    test_x, test_y = process_rows(test_data, glove_vectors)

    return np.array(train_x), np.array(test_x), np.array(train_y), np.array(test_y)

    for i, x in enumerate(train_x):
        if (i < 10):
            print("x", x)
            print("y", train_y[i])


def show_values(pc, fmt="%.2f", **kw):
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


mbti_index = {"ISTJ": 0, "ISFJ": 1, "INFJ": 2, "INTJ": 3, "ISTP": 4, "ISFP": 5, "INFP": 6, "INTP": 7, "ESTP": 8,
              "ESFP": 9, "ENFP": 10, "ENTP": 11, "ESTJ": 12, "ESFJ": 13, "ENFJ": 14, "ENTJ": 15}
d = {}
for key in mbti_index:
    d[mbti_index[key]] = key

mbti_labels = []
for i in range(16):
    mbti_labels.append(d[i])


def bidirectional_lstm(DATA_FILE, PERCENTAGE_TRAIN=0.7):
    print("mbti_rnn start!")
    train_x, test_x, train_y, test_y = prep_data(DATA_FILE, PERCENTAGE_TRAIN)

    # Layer's sizes
    x_size = train_x.shape[1]
    h_size = 50  # Number of hidden nodes
    y_size = train_y.shape[1]  # Number of outcomes
    batch_size = 64

    # Symbols
    # Shape: (batch_size, time, input_size)
    x = tf.placeholder(tf.float32, shape=[None, MAX_SEQ_LEN, GLOVE_DIMENSION])
    # Shape: (batch_size, num_outcomes)
    y = tf.placeholder(tf.float32, shape=[None, y_size])

    xavier_initializer = tf.contrib.layers.xavier_initializer()
    # W = tf.get_variable("W", shape=(h_size, y_size), initializer=xavier_initializer)
    # b = tf.get_variable("b", shape=(y_size), initializer=tf.constant_initializer(0))

    W = tf.get_variable("W", shape=(2 * h_size, y_size), initializer=xavier_initializer)
    b = tf.get_variable("b", shape=(y_size), initializer=tf.constant_initializer(0))

    # Re-shape data to work with RNNs, shape (time*batch_size, input_size)
    # x = tf.reshape(x, [-1, GLOVE_DIMENSION])
    # Split into 'time' tensors of shape (batch, input_size)
    # x = tf.split(axis=0, num_or_size_splits=MAX_SEQ_LEN, value=x)

    # lstm_fw_cell = rnn.BasicLSTMCell(h_size, forget_bias=1.0)
    # lstm_bw_cell = rnn.BasicLSTMCell(h_size, forget_bias=1.0)

    fw_cells = [rnn.BasicLSTMCell(h_size) for _ in range(batch_size)]
    bw_cells = [rnn.BasicLSTMCell(h_size) for _ in range(batch_size)]
    fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=1.0) for cell in fw_cells]
    bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=1.0) for cell in bw_cells]

    # x_cell = tf.unstack(x, MAX_SEQ_LEN, 1)

    # outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_cell,
    #                                              dtype=tf.float32)
    outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells, x,
                                                        dtype=tf.float32)

    attention_score = tf.nn.softmax(tf.contrib.slim.fully_connected(outputs, 1))
    attention_out = tf.squeeze(
        tf.matmul(tf.transpose(outputs, perm=[0, 2, 1]), attention_score),
        axis=-1)

    # cell = tf.nn.rnn_cell.BasicLSTMCell(h_size)
    # cell = tf.nn.rnn_cell.RNNCell
    # Output shape: (batch, time, output_size)
    # States shape: (batch, time, hidden_size) -> passed to next timestep

    # outputs, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    # outputs = tf.transpose(outputs, [0, 1, 2])
    #
    # last_output = outputs[-1]

    # lastOutput = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

    # yhat = tf.matmul(last_output, W) + b
    yhat = slim.fully_connected(attention_out, 16, activation_fn=tf.nn.sigmoid)

    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    lr = 0.1
    updates = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    print("Right before first epoch!")
    training_accuracies = []
    test_accuracies = []
    for epoch in range(500):
        # Train with each example
        for i in range(0, len(train_x), batch_size):
            sess.run(updates, feed_dict={x: train_x[i: i + batch_size], y: train_y[i: i + batch_size]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={x: train_x, y: train_y}))
        training_accuracies.append(train_accuracy)
        preds = sess.run(predict, feed_dict={x: test_x, y: test_y})
        test_accuracy = np.mean(np.argmax(test_y, axis=1) == preds)
        test_accuracies.append(test_accuracy)
        labels = []
        for i in range(len(test_y)):
            for j in range(len(test_y[i])):
                if test_y[i][j] == 1:
                    labels.append(j)
                    break
        print(np.array(labels), np.array(preds))
        cm = np.array([row / float(np.sum(row)) for row in confusion_matrix(np.array(labels), np.array(preds))])

        ### Display confusion matrix
        print("Confusion Matrix for Epoch: " + str(epoch))
        plt.rcParams["figure.figsize"] = (10, 10)
        heatmap = plt.pcolor(cm, cmap=plt.cm.Blues)
        # show_values(heatmap)

        plt.xticks(np.arange(len(mbti_labels)), mbti_labels)
        plt.yticks(np.arange(len(mbti_labels)), mbti_labels)
        plt.title("MBTI Prediction Confusion Matrix -- Epoch " + str(epoch + 1))
        # plt.show()

        plt.savefig('../data/rnn_lstm/' + str(epoch + 1) + ".png")
        plt.close()

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

        print(training_accuracies)
        print(test_accuracies)

    sess.close()


if __name__ == '__main__':
    INPUT_FILE = '../data/mbti_balanced_shuffled_data.txt'
    bidirectional_lstm(INPUT_FILE, 0.7)
