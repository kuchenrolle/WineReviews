import os
import sys

# need to ignore first gpu (0) on sever
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf

from time import time
from math import ceil

from model import Model, Phase
from config import DefaultConfig
from utils import DocumentProcessor, make_samples, Numberer

# number of topics lda looked for
num_topics = 50


def train_model(config, train_batches, validation_batch, use_topics = False, verbose = True, reuse = False):
    train_features, train_labels, train_seq_lens, train_word_lens, train_topics = train_batches
    validation_features, validation_labels, validation_seq_lens, validation_word_lens, validation_topics = validation_batch
#
    config.num_batches = train_features.shape[0]
    config.use_topics = use_topics
#
    with tf.Session() as sess:
        with tf.variable_scope("model", reuse = reuse):
            train_model = Model(
                config,
                phase = Phase.Train)
#
        with tf.variable_scope("model", reuse = True):
            validation_model = Model(
                config,
                phase = Phase.Validation)
#
        sess.run(tf.global_variables_initializer())
#
        for epoch in range(config.n_epochs):
            start = time()
            train_loss = 0.0
            validation_loss = 0.0
            accuracy = 0.0
#
            # Train on all batches.
            for batch in range(train_features.shape[0]):
                if verbose:
                    sys.stdout.write(f"\rTrain batch {batch+1} of {train_features.shape[0]}\r")
                loss, _ = sess.run([train_model.loss, train_model._train_op], {
                    train_model.x: train_features[batch], train_model.seq_lens: train_seq_lens[batch], train_model.word_lens: train_word_lens[batch], train_model.y: train_labels[batch], train_model.topics: train_topics[batch]})
                train_loss += loss
            if verbose:
                sys.stdout.write("\n")
            # validation
            for batch in range(validation_features.shape[0]):
                if verbose:
                    sys.stdout.write(f"\rvalidation batch {batch+1} of {validation_features.shape[0]}")
                loss, acc = sess.run([validation_model.loss, validation_model.accuracy], {
                    validation_model.x: validation_features[batch], validation_model.seq_lens: validation_seq_lens[batch], validation_model.word_lens: validation_word_lens[batch], validation_model.y: validation_labels[batch], validation_model.topics: validation_topics[batch]})
                validation_loss += loss
                accuracy += acc
#
            # normalize
            # validation_loss = validation_loss / config.validation_batch_size * config.batch_size
            accuracy /= validation_features.shape[0]
            accuracy *= 100
            train_loss /= train_features.shape[0]
            validation_loss /= validation_features.shape[0]
#
            took = time() - start
#
            if verbose:
                sys.stdout.write(f"\nepoch {epoch} - train loss: {train_loss:.2f}, validation loss: {validation_loss:.2f}, validation acc: {accuracy:.2f}, took: {took:.1f}s\n")


def test_and_generate(config, test_batches, character_indices, word_indices, num_reviews_to_produce, use_topics = False, verbose = True):
    test_features, test_labels, test_seq_lens, test_word_lens, test_topics = test_batches
    num_batches = test_features.shape[0]
    max_word_len = test_features.shape[-1]
    max_num_words = test_features.shape[-2]
    config.use_topics = use_topics
#
    with tf.Session() as sess:
        with tf.variable_scope("model", reuse = True):
            test_model = Model(
                config,
                phase = Phase.Validation)
        with tf.variable_scope("model", reuse = True):
            predict_model = Model(config)
#
        sess.run(tf.global_variables_initializer())
#
        test_loss = 0.0
        accuracy = 0.0
#
        for batch in range(num_batches):
            if verbose:
                sys.stdout.write(f"\rtest batch {batch+1} of {test_features.shape[0]}")
            loss, acc = sess.run([test_model.loss, test_model.accuracy], {
                test_model.x: test_features[batch], test_model.seq_lens: test_seq_lens[batch], test_model.word_lens: test_word_lens[batch], test_model.y: test_labels[batch], test_model.topics: test_topics[batch]})
            test_loss += loss
            accuracy += acc
#
        loss /= num_batches
        loss /= config.batch_size
        perplexity = 2**loss
        accuracy /= num_batches
#
        # produce sample reviews
        reviews = list()
        vocabulary = word_indices.names
#
        for _ in range(num_reviews_to_produce):
            choice = np.random.choice(range(test_topics.shape[1]))
            topic_distribution = test_topics[0][choice]
            topic_distribution = np.expand_dims(topic_distribution, axis = 0)
#
            review = ["<<<<"] # seed with beginning-of-review token "<<<<"
            arrays = [word2array("<<<<", character_indices, max_word_len)]
            seq_len = 1 # only seed token
            word_lens = [4] # seed token length
            next_word = ""
            num_sentences = 0
#
            while next_word != ">>>>" and num_sentences < 20 and len(review) < 500:
                features = arrays2features(arrays, max_num_words)
                word_lens_ = np.pad(word_lens[-seq_len:], [0, max_num_words - seq_len], mode = "constant")
                word_lens_ = np.expand_dims(word_lens_, axis = 0)
#
                [probabilities] = sess.run([predict_model.probs], {
                    predict_model.x: features, predict_model.seq_lens: np.array([seq_len]), predict_model.word_lens: word_lens_, predict_model.topics:topic_distribution})
                next_word = sample(vocabulary, probabilities[0][1:], config.temperature)
                word_lens.append(len(next_word))
                review.append(next_word)
                if next_word == ".":
                    num_sentences += 1
                arr = word2array(next_word, character_indices, max_word_len)
#
                arrays.append(arr)
                seq_len = min(seq_len+1, max_num_words)
#
                review_so_far = " ".join(review)
                sys.stdout.write(f"\r{review_so_far}")
#
            reviews.append(review)
#
    return accuracy, perplexity, reviews

# sample from probability distribution more or less conservatively
# temperature > 1 -> shift probability to more likely values
# temperature < 1 -> shift probability to less likely values
def sample(items, probabilities, temperature):
    logs = np.log(probabilities)
    logs *= temperature
    exps = np.exp(logs)
    probabilities = exps/np.sum(exps)
    item = np.random.choice(items, 1, p = probabilities)[0]
    return item

def word2array(word, character_indices, max_word_len):
    chars = list(word)
    indices = np.array([character_indices.number(char) for char in chars])
    array = np.pad(indices, [0, max_word_len - len(chars)], mode = "constant")
    return array

def arrays2features(arrays, max_num_words):
    num_words = len(arrays)
    max_num_chars = len(arrays[0])
    features = np.zeros((max_num_words, max_num_chars))
    start = max(0, num_words - max_num_words)
    for idx, array in enumerate(arrays[start:]):
        features[idx] = array
    features = np.expand_dims(features, axis = 0)
    return features

def to_dense(idxvals, len):
    vector = np.zeros(len)
    for idx, val in idxvals:
        vector[idx] = val
    return vector


if __name__ == "__main__":
np.random.seed(2311)
config = DefaultConfig()

# preprocessing
vkns = pd.read_pickle("data/vkns_with_topics.pkl")[["VKN","topics"]]
vkns = vkns.sample(frac = 0.05)
vkns = vkns.reset_index()
topics = vkns.topics
preprocessor = DocumentProcessor(vkns.VKN.values)

# make dense topic vectors
topics = vkns.topics.apply(to_dense, args = (num_topics,))

# process documents and get sample shape
max_word_len, min_doc_len, num_words, documents = preprocessor.get_corpus_statistics(return_processed_documents = True, only_most_frequent = config.only_most_frequent)

# make samples
samples_dict = make_samples(documents, topics = topics, num_samples = num_words, max_steps = config.max_timesteps, max_word_len = max_word_len, only_most_frequent = config.only_most_frequent)

# get indices
word_indices = samples_dict["words"]
character_indices = samples_dict["characters"]

# get shapes for training, validation and test data
train = int(num_words * 0.85 // config.batch_size * config.batch_size)
train_shape = (train//config.batch_size, config.batch_size, config.max_timesteps, max_word_len)

test = int(num_words * 0.95 // config.batch_size * config.batch_size)
test_shape = ((test-train)//config.batch_size, config.batch_size, config.max_timesteps, max_word_len)

validation_shape = (ceil((num_words - test) / config.batch_size), config.batch_size, config.max_timesteps, max_word_len)


# shape features
train_features = samples_dict["samples"][:train].reshape(train_shape)
test_features = samples_dict["samples"][train:test].reshape(test_shape)
validation_features = samples_dict["samples"][test:].copy()
validation_features.resize(validation_shape) # pad with zeros

# shape labels
train_labels = samples_dict["labels"][:train].reshape(train_shape[:2])
test_labels = samples_dict["labels"][train:test].reshape(test_shape[:2])
validation_labels = samples_dict["labels"][test:].copy()
validation_labels.resize(validation_shape[:2])

# shape topic distributions
train_topics = samples_dict["topics"][:train].reshape((train_shape[0], train_shape[1], num_topics))
test_topics = samples_dict["topics"][train:test].reshape((test_shape[0], test_shape[1], num_topics))
validation_topics = samples_dict["topics"][test:].copy()
validation_topics.resize((validation_shape[0],validation_shape[1],num_topics))

# shape actual sequence lengths
train_seq_lens = samples_dict["sequence_lengths"][:train].reshape(train_shape[:2])
test_seq_lens = samples_dict["sequence_lengths"][train:test].reshape(test_shape[:2])
validation_seq_lens = samples_dict["sequence_lengths"][test:].copy()
validation_seq_lens.resize(validation_shape[:2])

# shape actual word lengths
train_word_lens = samples_dict["word_lengths"][:train].reshape(train_shape[:3])
test_word_lens = samples_dict["word_lengths"][train:test].reshape(test_shape[:3])
validation_word_lens = samples_dict["word_lengths"][test:].copy()
validation_word_lens.resize(validation_shape[:3])

# pack together
train_batches = train_features, train_labels, train_seq_lens, train_word_lens, train_topics
test_batches = test_features, test_labels, test_seq_lens, test_word_lens, test_topics
validation_batch = validation_features, validation_labels, validation_seq_lens, validation_word_lens, validation_topics

# complete config
config.max_word_len = max_word_len
config.num_labels = len(word_indices.names) + 1
config.num_chars = len(character_indices.names) + 1
config.num_topics = num_topics


# train the model
train_model(config, train_batches, validation_batch)
# train_model(config, train_batches, train_batches)

# test the model
accuracy, perplexity, reviews = test_and_generate(config,
                test_batches,
                character_indices,
                word_indices,
                config.num_reviews_to_produce)

# reset
tf.reset_default_graph()

# now train with topic distributions
train_model(config, train_batches, validation_batch, use_topics = True)

# and test
accuracy_topics, perplexity_topics, reviews_topics = test_and_generate(config,
                test_batches,
                character_indices,
                word_indices,
                config.num_reviews_to_produce,
                use_topics = True)

