import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re

import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

lyrics2vec = w2v.Word2Vec.load(os.path.join(
    "trained", "l222vec800_500feat_0downsample.w2v"))

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = lyrics2vec.wv.vectors
#all_word_vectors_matrix_2d = np.load("trained\l222vec800_500feat_0downsample.w2v.wv.vectors.npy")
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
np.save("matrix-final", all_word_vectors_matrix_2d)
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[lyrics2vec.wv.vocab[word].index])
            for word in lyrics2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)


sns.set_context("notebook")
points.plot.scatter("x", "y", s=10, figsize=(100, 100))
plt.savefig('foo.png')
'''
def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]

    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

plot_region(x_bounds=(4.0, 4.5), y_bounds=(-0.5, -0.1))

plot_region(x_bounds=(-0.8, 0.8), y_bounds=(-0.8, 0.8))
'''
print(lyrics2vec.most_similar("weed"))


def nearest_similarity_cosmul(start1, end1, end2):
    similarities = lyrics2vec.wv.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(
        **locals()))
    return start2

nearest_similarity_cosmul("blood", "gun", "red")
nearest_similarity_cosmul("rhyme", "sword", "gun")
nearest_similarity_cosmul("crip", "shit", "dragons")
