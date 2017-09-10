#!/usr/bin/python3
import sys
import spacy

import pandas as pd
import numpy as np

from math import ceil
from time import time

from collections import Counter
from nltk.corpus import stopwords


class SnippetTimer:

  def __enter__(self):
    self.start = time()

  def __exit__(self, *args):
    self.end = time()
    took = self.end - self.start
    sys.stdout.write(f"Snippet took {took} seconds!\n")


class Numberer:

    def __init__(self):
        self._known = dict()
        self._items = list()
        self._current = 0
        self._add = True


    def number(self, item):
        idx = self._known.get(item)
        if idx is None:
            if self._add:
                self._current += 1
                self._items.append(item)
                self._known[item] = self._current
                return self._current
            else:
                return 0
        else:
            return idx

    def name(self, idx):
        if idx < 1:
            return "UNKNOWN"
        return self._items[idx-1]

    def freeze(self):
        self._add = False

    def unfreeze(self):
        self._add = True

    @property
    def names(self):
        return self._items

    @property
    def to_dict(self):
        return self._known

    @property
    def to_reverse_dict(self):
        return {idx:item for item, idx in self._known.items()}


class DocumentProcessor:

  def __init__(self, documents,
              language = "de",
              lowercase = True,
              to_indices = False,
              bag_of_words = False,
              remove_stopwords = False,
              remove_punctuation = False,
              pos_to_filter = set(),
              add_start_end_token = True,
              max_word_len = 25):

    self._documents = documents
    self._nlp = spacy.load(language)
    self._lowercase = lowercase
    self._bag_of_words = bag_of_words
    self._pos_to_filter = pos_to_filter
    self._add_start_end = add_start_end_token
    self._to_indices = to_indices
    self._remove_stopwords = remove_stopwords
    self._max_word_len = max_word_len

    if remove_punctuation:
      pos_to_filter.add("PUNCT")
    
    if remove_stopwords:
      if language == "de":
        self._stopwords = set(stopwords.words("german"))
      elif language == "en":
        self._stopwords = set(stopwords.word("english"))
    if to_indices:
      self._indices = Numberer()


  def __iter__(self):
    documents = self._documents
    for document in documents:
      yield self._process_document(document)


  def get_indices(self):
    for document in self._documents:
      _ = self._process_document(document)
    return self._indices.to_reverse_dict


  def get_corpus_statistics(self, return_processed_documents = False, only_most_frequent = False):
    documents = [self._process_document(doc) for doc in self._documents]

    if only_most_frequent:
      word_frequencies = Counter()
      _ = [word_frequencies.update(doc) for doc in documents]
      most_frequent = [word for word, _ in word_frequencies.most_common(only_most_frequent)]    

    processed = []
    max_word_len = num_words = 0
    min_doc_len = 100
    for num_documents, doc in enumerate(documents):
      max_word_len_ = max([len(word) for word in doc])
      max_word_len = max(max_word_len, max_word_len_)
      min_doc_len = min(min_doc_len, len(doc))
      num_words += len(doc)
      if return_processed_documents:
        processed.append(doc)
    if return_processed_documents:
      return max_word_len, min_doc_len, num_words, processed
    else:
      return max_word_len, min_doc_len, num_words


  def _process_document(self, document):
    document = document.replace("\\r\\n", "\n")
    document = document.replace("\\n", "\n")
    processed = self._nlp(document)
    
    words = list()
    if self._add_start_end:
      words.append("<<<<")
    for sentence in processed.sents:
      for word in sentence:
        # replace urls
        if word.like_url:
          words.append("http://bit.ly/23Chr11Ad")
          continue
        # filter pos tags
        if word.pos_ in self._pos_to_filter:
          continue
        # drop long words
        if len(word.orth_) < self._max_word_len:
          words.append(word.orth_)


    if self._lowercase:
      words = [word.lower() for word in words]
    if self._remove_stopwords:
      words = [word for word in words if word not in self._stopwords]
    if self._add_start_end:
      words.append(">>>>")
    if self._to_indices:
      words = [self.index(word) for word in words]
    if self._bag_of_words:
      words = list(Counter(words).items())

    return words
    

  def index(self, word):
    return self._indices.number(word)


# if first word is not to be predicted, num_samples should be changed,
# otherwise the last num_documents rows of the output are empty
def make_samples(documents, topics, num_samples, max_steps, max_word_len, dont_predict_first_words = True, only_most_frequent = None):
    num_topics = len(topics[0])
    one_or_zero = dont_predict_first_words
    word_indices = Numberer()
    character_indices = Numberer()
#
    samples = np.zeros((num_samples, max_steps, max_word_len), dtype = np.int32)
    labels = np.zeros((num_samples), dtype = np.int32)
    tops = np.zeros((num_samples, num_topics), dtype = np.int32)
    sequence_lengths = np.zeros((num_samples), dtype = np.int32)
    word_lengths = np.zeros((num_samples, max_steps), dtype = np.int32)
#
    if only_most_frequent:
      word_frequencies = Counter()
      _ = [word_frequencies.update(doc) for doc in documents]
      for word,_ in word_frequencies.most_common(only_most_frequent):
        _ = word_indices.number(word)
      word_indices.freeze()
#
    offset = 0
    for doc_id, document in enumerate(documents):
      num_terms = len(document)
      lengths_ = np.array([min(x, max_steps) for x in range(num_terms)], dtype = np.int32)
#
      for idx in range(one_or_zero,num_terms):
        sequence = document[max(0, idx - lengths_[idx]):idx]
        sequence = [np.array([character_indices.number(char) for char in word]) for word in sequence]
        word_lengths_ = np.array([len(word) for word in sequence])
        for timestep in range(len(sequence)):
          sample = idx+offset-one_or_zero
          wordlen = word_lengths_[timestep]
          samples[sample][timestep][:wordlen] = sequence[timestep]
        word_lengths[sample][:len(sequence)] = word_lengths_
#
      labels_ = [word_indices.number(word) for word in document]
      tops_ = [topics[doc_id] for w in document]
      labels[offset:offset+num_terms-one_or_zero] = labels_[one_or_zero:]
      tops[offset:offset+num_terms-one_or_zero] = tops_[one_or_zero:]
      sequence_lengths[offset:offset+num_terms-one_or_zero] = lengths_[one_or_zero:]
      offset += num_terms - one_or_zero
#
    output = {}
#
    output["samples"] = samples
    output["labels"] = labels
    output["topics"] = tops
    output["sequence_lengths"] = sequence_lengths
    output["word_lengths"] = word_lengths
    output["words"] = word_indices
    output["characters"] = character_indices
#
    return output


