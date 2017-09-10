#!/usr/bin/python3
import csv

import pandas as pd
import numpy as np

from utils import DocumentProcessor
from gensim.models import LdaMulticore

min_frequency = 50
num_topics = 50


# changes columns in dataframe to category with labels
def categorize_column(df, category_label_pair):
  category, labels = category_label_pair
  labels = pd.Index(labels)
  df[category] = df[category].astype("category")
  df[category].cat.categories = labels


# line 23378 (23377) throws a parsing error
# because of a single quote, set
# quoting accordingly
vkns = pd.read_csv(filepath_or_buffer = "data/vkn_export08082017.txt",
                  encoding = "latin-1",
                  sep = "\t",
                  # skiprows = [23377],
                  quoting = csv.QUOTE_NONE)

# drop uninteresting columns
uninteresting = ["Autor","Bewertung","TrinkreifeText","Flaschengr"]
vkns.drop(uninteresting, inplace = True, axis = 1)

# translate 20pt system to 100pt
# and cleanup
indices = vkns.Punkte[vkns.Punkte <= 20].index
new_values = vkns.Punkte.loc[indices] * 2.5 + 50
vkns.Punkte.loc[indices] = new_values
vkns.Punkte.loc[(vkns.Punkte > 100) | (vkns.Punkte < 50)] = 0
vkns.Punkte = vkns.Punkte.round()

# get column types and names right
# integers
int_columns = ["ID","Jahrgang","Punkte"]
vkns[int_columns] = vkns[int_columns].fillna(0).astype(int)

# categories with labels already given
categories = ["Produzent", "Rebsorte", "Land", "Region", "UserID"]
for category in categories:
  vkns[category] = vkns[category].astype("category")

# categories with external labels
categories_with_labels = [("Typ", ["Rotwein", "Weißwein", "Rosé", "Schaumwein", "Süßwein", "Gespriteter Wein"]),
("Trinkreife",["keine Angabe", "noch lagern", "trinken oder lagern", "jetzt trinken","schon abbauend"]),
("PLV", ["keine Angabe", "schlecht", "akzeptabel/angemessen", "gut", "grandios"]),
("Verschluss", ["keine Angabe", "Naturkork", "Kunststoff", "Schraubverschluss", "Glasstopfen", "Kronkorken", "sonstige"]),
("Bezugsquelle", ["Hof","Handel","Subskription","Abverkauf/Sonderangebot", "Auktion", "Sonstige Bezugsquelle"])]
for category_label_pair in categories_with_labels:
  categorize_column(vkns, category_label_pair)


# remove empty reviews
vkns = vkns[vkns.VKN.notnull()]

# process into documents
documents = DocumentProcessor(vkns.VKN,
              to_indices = True,
              bag_of_words = True,
              remove_stopwords = True,
              remove_punctuation = True,
              add_start_end_token = False)

# get mapping from indices to words
mapping = documents.get_indices()

# run lda
lda = LdaMulticore(corpus = documents,
                passes = 10,
                id2word = mapping,
                num_topics = num_topics,
                workers = 16)

# get topic distribution for each document
topic_distributions = [lda[document] for document in documents]
# add results to data frame
vkns["topics"] = pd.Series(topic_distributions).values

# save
vkns.to_pickle("data/vkns_with_topics.pkl")
