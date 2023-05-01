import gensim.models
from gensim.test.utils import datapath
from gensim import utils
import unicodedata
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        corpus_path = datapath(self.data_path)
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


def generate_model(data_path, model_path=None):
    data = MyCorpus(data_path)

    model = gensim.models.Word2Vec(sentences=data)

    if model_path is not None:
        model.save(model_path)


def load_model(model_path):
    return gensim.models.Word2Vec.load(model_path)


def find_similar_words(*models, remove_accent=True):
    dictionary = {}

    for i in range(len(models)):
        for item in models[i].wv.index_to_key:
            if remove_accent:
                processed_item = remove_accents(item)
            else:
                processed_item = item
            if processed_item in dictionary.keys():
                dictionary[processed_item] += f"{i}"
            else:
                dictionary[processed_item] = f"{i}"

    words = []
    for key, value in dictionary.items():
        suc = True
        for i in range(len(models)):
            if f"{i}" not in value:
                suc = False
        if suc:
            words.append(key)

    return words


def load_similar_words():
    words = open("similar_words.txt", "r")

    return words.read().splitlines()


def find_word(word, model):
    potential_words = model.wv.index_to_key

    for potential_word in potential_words:
        if remove_accents(potential_word) == word:
            return potential_word

    return word


def get_embeddings(model, words):
    embeddings = {}
    count = 0
    for word in words:
        real_word = find_word(word, model)
        if real_word in model.wv:
            embeddings[word] = model.wv[real_word]
            count += 1
    return embeddings


def compute_relation_vectors(embeddings):
    relation_vectors = {}
    for word, emb in embeddings.items():
        diff_vectors = []
        for other_word, other_emb in embeddings.items():
            if word != other_word:
                diff_vectors.append(emb - other_emb)
        relation_vectors[word] = np.mean(diff_vectors, axis=0)
    return relation_vectors

def normalize_relation_vectors(relation_vectors):
    normalized_relation_vectors = {}
    for word, vec in relation_vectors.items():
        normalized_relation_vectors[word] = vec / np.linalg.norm(vec)
    return normalized_relation_vectors


def compare_relation_vectors(vecs1, vecs2):
    similarities = {}
    for word in vecs1.keys():
        vec1 = vecs1[word].reshape(1, -1)
        vec2 = vecs2[word].reshape(1, -1)
        similarity = cosine_similarity(vec1, vec2)[0][0]
        similarities[word] = similarity
    return similarities

def visualize_embeddings(model, words_to_label):
    vectors = []
    included_words =  []
    for w in common_words:
        try:
            vectors.append(model.wv[find_word(w, model)])
            included_words.append(w)
        except: continue

    vectors = np.array(vectors)
    pca = PCA(n_components=2)

    pca_features = pca.fit_transform(vectors)
    pca_df = pd.DataFrame(
        data=pca_features, 
        columns=['PC1', 'PC2'])
    pca_df['word'] = included_words

    sns.set()
    sns.lmplot(x='PC1', 
        y='PC2', 
        data=pca_df, 
        fit_reg=False, 
        legend=True
    )

    for _, row in pca_df.iterrows():
        if row['word'] not in words_to_label: continue
        x = row['PC1']
        y = row['PC2']
        plt.text(x * (1 + 0.01), y * (1 + 0.01) , row['word'], fontsize=9)

    plt.title('2D PCA')
    plt.show()


if __name__ == "__main__":
    french_model = load_model("gensim-model-french")
    haitian_model = load_model("gensim-model-haitian-creole-combined")
    mauritian_model = load_model("gensim-model-mauritian-creole")
    common_words = load_similar_words()

    embeddings_haitian_creole = get_embeddings(haitian_model, common_words)
    embeddings_mauritian_creole = get_embeddings(mauritian_model, common_words)
    embeddings_french = get_embeddings(french_model, common_words)

    relation_vectors_haitian_creole = compute_relation_vectors(embeddings_haitian_creole)
    relation_vectors_mauritian_creole = compute_relation_vectors(embeddings_mauritian_creole)
    relation_vectors_french = compute_relation_vectors(embeddings_french)

    normalized_relation_vectors_haitian_creole = normalize_relation_vectors(relation_vectors_haitian_creole)
    normalized_relation_vectors_mauritian_creole = normalize_relation_vectors(relation_vectors_mauritian_creole)
    normalized_relation_vectors_french = normalize_relation_vectors(relation_vectors_french)

    similarities_haitian_mauritian = compare_relation_vectors(normalized_relation_vectors_haitian_creole,
                                                              normalized_relation_vectors_mauritian_creole)
    similarities_haitian_french = compare_relation_vectors(normalized_relation_vectors_haitian_creole,
                                                           normalized_relation_vectors_french)
    similarities_mauritian_french = compare_relation_vectors(normalized_relation_vectors_mauritian_creole,
                                                             normalized_relation_vectors_french)

    sorted_similarities_haitian_mauritian = sorted(similarities_haitian_mauritian.items(), key=lambda x: x[1], reverse=True)
    sorted_similarities_haitian_french = sorted(similarities_haitian_french.items(), key=lambda x: x[1], reverse=True)
    sorted_similarities_mauritian_french = sorted(similarities_mauritian_french.items(), key=lambda x: x[1], reverse=True)

    print("Similarities between Haitian Creole and Mauritian Creole:", sorted_similarities_haitian_mauritian)
    print("Similarities between Haitian Creole and French:", sorted_similarities_haitian_french)
    print("Similarities between Mauritian Creole and French:", sorted_similarities_mauritian_french)




