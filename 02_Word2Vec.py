# 9. Use a prebuilt Word2vec model to find the embedding of the given word. 
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import gensim.downloader

model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)


model.save("word2vec.model")

model = Word2Vec.load("word2vec.model")

vector = model.wv['computer'] # get numpy vector of a word
vector

similar = model.wv.most_similar('computer', topn=10)  # get other similar words
similar


vector1 = model.wv['user'] 
vector1

similar1 = model.wv.most_similar('user', topn=10)  # get other similar words
similar1

gensim_pretrained_models = list(gensim.downloader.info()['models'].keys())
for model in gensim_pretrained_models:
    print(model)
#print([model in gensim_pretrained_miodels])

word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')

word2vec_vectors.most_similar('google', topn=10)

sims = word2vec_vectors["king"] - word2vec_vectors["man"] + word2vec_vectors["woman"]
word2vec_vectors.most_similar([sims])

sims = word2vec_vectors["Messi"] - word2vec_vectors["football"] + word2vec_vectors["cricket"]
word2vec_vectors.most_similar([sims])

