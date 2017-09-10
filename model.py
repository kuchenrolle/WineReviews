from enum import Enum

import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn


class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2


class Model:
    def __init__(
            self,
            config,
            phase = Phase.Predict,):

        self._phase = phase
        
        num_batches = config.num_batches
        num_topics = config.num_topics
        batch_size = config.batch_size
        if phase == Phase.Predict:
            batch_size = 1

        input_len = config.max_timesteps
        input_size = config.max_word_len
        vocabulary_size = config.num_labels
        num_chars = config.num_chars
        
        char_embedding_length = config.char_embedding_length
        word_embedding_length = config.word_embedding_length

        # dropout only during training
        if self.phase == Phase.Train:
            input_dropout = config.input_dropout
            hidden_dropout = config.hidden_dropout
        else:
            input_dropout = hidden_dropout = 1.0


        # The integer-encoded character of the words. input_len is the
        # maximum number of time steps, input_size is the maximum
        # number of characters per word
        self._x = tf.placeholder(tf.int32, shape = [batch_size, input_len, input_size])

        # These tensors provide the actual number of time steps for each
        # sequence and characters for each word
        self._seq_lens = tf.placeholder(tf.int32, shape = [batch_size])
        self._word_lens = tf.placeholder(tf.int32, shape = [batch_size, input_len])

        # words to predict
        if self.phase != Phase.Predict:
            self._y = tf.placeholder(
                tf.int32, shape=[batch_size])
            y_one_hot = tf.one_hot(self._y, depth = vocabulary_size)

        # topic distributions
        self._topics = tf.placeholder(tf.float32, shape = [batch_size, num_topics])

        # Embeddings
        self.embeddings = tf.get_variable("character_embeddings", [num_chars, char_embedding_length])

        # get character embeddings and reshape to words rather than sequences
        x_embedded = tf.nn.embedding_lookup(params = self.embeddings, ids = self.x)
        x_embedded = tf.reshape(x_embedded, shape = [-1, input_size, char_embedding_length])
        x_embedded = tf.nn.dropout(x_embedded, keep_prob = input_dropout)

        # word lengths needs to be reshaped as well
        word_lens = tf.reshape(self.word_lens, shape = [-1])


        # bidirectional stacked GRU
        cells = []

        with tf.variable_scope("character_rnn"):
            for i in range(config.num_layers_char*2):
                cell = rnn.GRUCell(config.cell_size_char)
                cell = rnn.DropoutWrapper(cell, output_keep_prob = hidden_dropout, variational_recurrent = True, dtype = tf.float32)
                cells.append(cell)

            forward_cell_char = rnn.MultiRNNCell(cells[:config.num_layers_char], state_is_tuple = False)
            backward_cell_char = rnn.MultiRNNCell(cells[config.num_layers_char:], state_is_tuple = False)

            _, states = tf.nn.bidirectional_dynamic_rnn(forward_cell_char, backward_cell_char, x_embedded, dtype = tf.float32, sequence_length = word_lens)
            states = tf.concat(states, -1)

        # turn gru output into word embeddings and reshape
        logits = tf.layers.dense(states, word_embedding_length)
        x = tf.reshape(logits, shape = [batch_size, input_len, word_embedding_length])

        # now run words through GRU
        cells = []

        with tf.variable_scope("word_rnn"):
            for i in range(config.num_layers_word*2):
                cell = rnn.GRUCell(config.cell_size_word)
                cell = rnn.DropoutWrapper(cell, output_keep_prob = hidden_dropout, variational_recurrent = True, dtype = tf.float32)
                cells.append(cell)

            forward_cell = rnn.MultiRNNCell(cells[:config.num_layers_word], state_is_tuple = False)
            backward_cell = rnn.MultiRNNCell(cells[config.num_layers_word:], state_is_tuple = False)

            _, states = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, x, dtype = tf.float32, sequence_length = self.seq_lens)
            states = tf.concat(states, -1)

        # use topic distributions
        if config.use_topics:
            states = tf.concat([states, self._topics], -1)

        # finally get output distribution
        logits = tf.layers.dense(states, vocabulary_size)
        
        if self.phase != Phase.Predict:
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels = y_one_hot, logits = logits)
            self._loss = loss = tf.reduce_sum(losses)

        if self.phase == Phase.Train:
            start_lr = config.learning_rate
            global_step = tf.Variable(0, trainable = False)
            learning_rate = tf.train.exponential_decay(start_lr, global_step, num_batches, config.decay_rate)

            self._train_op = tf.train.AdamOptimizer(learning_rate).minimize(losses, global_step = global_step)
            self._probs = probs = tf.nn.softmax(logits)

        if self.phase == Phase.Validation:
            # Highest probability labels of the gold data.
            hp_labels = tf.argmax(y_one_hot, axis = 1)

            # Predicted labels
            labels = tf.argmax(logits, axis = 1)

            correct = tf.equal(hp_labels, labels)
            correct = tf.cast(correct, tf.float32)
            self._accuracy = tf.reduce_mean(correct)

        if self.phase == Phase.Predict:
            # probability distribution over next word
            self._probs = tf.nn.softmax(logits)

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def loss(self):
        return self._loss

    @property
    def seq_lens(self):
        return self._seq_lens

    @property
    def word_lens(self):
        return self._word_lens

    @property
    def probs(self):
        return self._probs

    @property
    def train_op(self):
        return self._train_op

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def topics(self):
        return self._topics

    @property
    def phase(self):
        return self._phase