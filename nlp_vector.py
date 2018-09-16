from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec
import collections
import pickle

class Base_Model(object):
    def __init__(self):
        self.sample_vecs = dict()
        self.text_dict = dict()
        self.vector_size = None
        self.similaritys=[]

    def cosine(self, vector1, vector2):
        numerator = np.dot(vector1, vector2)
        denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return numerator / denominator

    def averaged_doc(self, model_wv):
        self.sample_vecs = {}
        for key, items in self.text_dict.items():
            temp = np.zeros(self.vector_size)
            n = 0
            for item in items:
                try:  # 防止出现:要的词不在词典里
                    temp = temp + model_wv(item)
                    n = n + 1
                except:
                    temp = temp
            if n != 0:
                temp = temp / n
            else:
                print('%s is unbelievable' % key)
                pass
            self.sample_vecs[key] = temp

    def top_N(self, vec, N):
        self.similaritys = []
        for key in self.sample_vecs.keys():
            similarity = self.cosine(vec, self.sample_vecs[key])
            self.similaritys.append(similarity)
        top_N = np.argsort(-np.array(self.similaritys))[:N]
        result = []
        for item in top_N:
            result.append(list(self.sample_vecs.keys())[item])
        return result

    def save(self,data, name):
        with open("./Model/%s" % name + ".pickle", 'wb') as f:
            pickle.dump(data, f)

    def load(self,name):
        with open("./Model/%s" % name + ".pickle", 'rb') as f:
            return pickle.load(f)

class TFIDF_Model(Base_Model):
    """
    parameters
    -----------
    keys: keys for every text
    texts: list of strs
        e.g. ['oboz women’s bridger b-dry hiking boot','ivanka trump women’s kayden pump']
    text: list of str
        e.g. ['ivanka trump women’s kayden pump']

    returns
    ---------------
    self.feature: all words in vocabulary
    self.text_dict: (key,value) for all input texts; value is raw text; key is corresponding asin
    self.sample_vecs: dict for (key,value) for all texts vectors; value is text vector; key is corresponding asin
    """

    def __init__(self):
        super().__init__()  # 等同于super(TFIDF_Model, self).__init__()
        self.model = None
        self.feature = None
        self.matrix=None

    def build_model(self, texts, keys):
        self.sample_vecs = dict()
        self.model = TfidfVectorizer(stop_words="english")
        self.matrix = self.model.fit_transform(texts).toarray()
        self.feature = self.model.get_feature_names()
        for i, key in enumerate(keys):
            self.text_dict[key] = texts[i]
            self.sample_vecs[key] = self.matrix[i]

    def top_N(self, text, N):
        vec = self.model.transform(text).toarray()[0]
        return Base_Model.top_N(self, vec, N)


class Word2Vec_Model(Base_Model):
    """
    parameters
    ---------------
    keys: keys for every text
    texts: list of strs
        e.g. ['oboz women’s bridger b-dry hiking boot','ivanka trump women’s kayden pump']
    text: list of str
        e.g. ['ivanka trump women’s kayden pump']

    returns
    ---------------
    self.text_dict: (key,value) for all input texts; value is raw text; key is corresponding asin
    self.sample_vecs: dict for (key,value) for all texts vectors; value is text vector; key is corresponding asin
    """

    def build_model(self, texts, keys, size=100, min_count=3, window=5):
        """
        parameters
        -------------
        window: scan window
        min_count: minimal count number (if the frequency of the word <min_count, the word is ignore )
        """
        self.text_dict = dict()
        texts = [sentence.split() for sentence in texts]
        for i, key in enumerate(keys):
            self.text_dict[key] = texts[i]
        self.vector_size = size
        min_count = min_count
        window = window
        self.model = Word2Vec(size=self.vector_size, min_count=min_count, window=window)
        self.model.build_vocab(texts)
        self.model.train(texts, total_examples=self.model.corpus_count, epochs=self.model.iter)

    def averaged_doc(self):
        Base_Model.averaged_doc(self, self.model.wv.get_vector)

    def top_N(self, text, N):
        text = text[0].split()
        n = 0
        temp = np.zeros(self.vector_size)
        for item in text:
            try:  # 防止出现:要的词不在词典里
                temp = temp + self.model.wv.get_vector(item)
                n = n + 1
            except:
                temp = temp
        if n != 0:
            temp = temp / n
        else:
            print('the result is unbelievable')
            pass
        vec = temp
        return Base_Model.top_N(self, vec, N)


from scipy import sparse
from functools import wraps


def listfunc(func):
    @wraps(func)
    def op(*arg, **args):
        result = collections.defaultdict(list)
        for row, col, value in func(*arg, **args):
            result[row].append((col, value))
        return result
    return op


class Glove_Model(Base_Model):
    """
    parameters
    ---------------
    keys: keys for every text
    texts: list of strs
        e.g. ['oboz women’s bridger b-dry hiking boot','ivanka trump women’s kayden pump']
    text: list of str
        e.g. ['ivanka trump women’s kayden pump']

    references
    ---------------
    1.Jeffrey Pennington, Richard Socher, Christopher D. Manning
        GloVe: Global Vectors for Word Representation.
    2.https://www.cnblogs.com/Weirping/p/7999979.html
    """

    def __init__(self):
        super().__init__()
        self.id2word = None
        self.word2id = None
        self.model_wv = None

    @listfunc
    def build_model(self, texts, keys, window=5, min_count=3):
        """
        parameters
        -------------
        window: scan window (window=5 means the left window of the center word is 5, so does the right window)
        min_count: minimal count number (if the frequency of the word <min_count, the word is ignore )
        """
        self.text_dict = dict()
        texts = [sentence.split() for sentence in texts]
        vocab = collections.Counter()
        for i, key in enumerate(keys):
            self.text_dict[key] = texts[i]
            vocab.update(texts[i])
        self.word2id = {word: i for i, word in enumerate(vocab)}
        self.id2word = {i: word for i, word in enumerate(vocab)}
        matrix = sparse.lil_matrix((len(self.word2id), len(self.word2id)), dtype=np.float64)
        for text in texts:
            content_token = [self.word2id[word] for word in text]
            for center_i, center_id in enumerate(content_token):
                left = max(0, center_i - window)
                context = content_token[left:center_i]
                for context_i, context_id in enumerate(context):
                    w = 1.0   #1.0/(center_i-context_i-left)
                    matrix[center_id, context_id] += w
                    matrix[context_id, center_id] += w
        for row, (j, value) in enumerate(zip(matrix.rows, matrix.data)):
            for col, weight in zip(j, value):
                if weight >= min_count:
                    yield row, col, weight

    def train(self, co_matrix, vector_size=100, iteration=100, learning_rate=0.01, eps=0.00001, xmax=100, alpha=0.75):
        """
        loss function: sum(f(Xij)*(wi*wj+bi+bj-log(1+Xij))^2)
        f(Xij)=(X/Xmax)^alpha  if X<Xmax
                      1         otherwise
        batch ALS for training
        """
        np.random.seed(1)
        feature_size = len(self.word2id)
        self.vector_size = vector_size

        # w=np.random.normal(1,0.1,size=(feature_size,vector_size))
        # bias=np.random.normal(1,0.1,size=feature_size)
        w = (np.random.rand(feature_size, vector_size) - 0.5) / float(vector_size + 1)
        bias = (np.random.rand(feature_size) - 0.5) / float(vector_size + 1)
        for i in range(iteration):
            previous_w = w.copy()
            previous_b = bias.copy()
            for row in co_matrix.keys():
                gradient_w = 0;
                gradient_b = 0
                for col, value in co_matrix[row]:
                    f_xij = (value / xmax) ** alpha if value < xmax else 1
                    cost_ = w[row].dot(w[col]) + bias[row] + bias[col] - np.log( value)
                    gradient_w += f_xij * cost_ * w[col]
                    gradient_b += f_xij * cost_
                    # gradient_w=f_xij*cost_*w[col]
                    # gradient_b = f_xij * cost_
                w[row] = w[row] - learning_rate * gradient_w
                bias[row] = bias[row] - learning_rate * gradient_b
            if abs(np.linalg.norm(w - previous_w)) < eps \
                    and abs(np.linalg.norm(bias) - np.linalg.norm(previous_b)) < eps:
                print("stop iter:", i)
                break
        self.model_wv = dict()
        for key in co_matrix.keys():
            self.model_wv[self.id2word[key]] = w[key]

    def averaged_doc(self):
        Base_Model.averaged_doc(self, self.model_wv.get)

    def top_N(self, text, N):
        text = text[0].split()
        n = 0
        temp = np.zeros(self.vector_size)
        for item in text:
            try:  # 防止出现:要的词不在word2vec词典里
                temp = temp + self.model_wv.get(item)
                n = n + 1
            except:
                temp = temp
        if n != 0:
            temp = temp / n
        else:
            print('the result is unbelievable')
            pass
        vec = temp
        return Base_Model.top_N(self, vec, N)


class LDA_Model(Base_Model):
    """
    gibbs sampling for training LDA model

    references
    ----------
    1.book : LDA数学八卦
    """

    def __init__(self):
        super().__init__()
        self.word2id = None
        self.id2word = None
        self.W_T = None
        self.D_T = None

    def build_model(self, texts, keys, min_count=3):
        """
        dirichlet distribution:
            document_topics:1/delta(alpha)*x1^(alpha1-1)x2^(alpha2-1)
            topic_words=1/delta(beta)*x1^(beta1-1)x2^(beta2-1)
        """
        self.text_dict = dict()
        vocab = collections.Counter()
        texts = [sentence.split() for sentence in texts]
        for i, key in enumerate(keys):
            self.text_dict[key] = texts[i]
            vocab.update(texts[i])
        filter_vocab = [key for key, value in vocab.items() if value >= min_count]
        self.word2id = {word: i for i, word in enumerate(filter_vocab)}
        self.id2word = {i: word for i, word in enumerate(filter_vocab)}

    def train(self, topic_num=100, alpha=2, beta=2, iteration=500):
        """
        parameters
        --------------
        topic_num: the same as vector length

        according the formula about P50 of the book LDA数学八卦
        #the eps if you set should adjust according to the number of words
        """
        ## init all variables
        topic_array = []
        document_array = []
        words_array = []

        for doc_i, text in enumerate(self.text_dict.values()):
            for word in text:
                if word in self.word2id.keys():
                    topic_array.append(np.random.randint(0, topic_num))
                    document_array.append(doc_i)
                    words_array.append(self.word2id[word])
        topic_array = np.array(topic_array)
        document_array = np.array(document_array)
        words_array = np.array(words_array)

        self.D_T = np.zeros((len(self.text_dict.values()), topic_num))
        self.W_T = np.zeros((len(self.word2id.keys()), topic_num))
        for doc_i in range(len(self.text_dict.values())):
            topics = topic_array[np.where(document_array == doc_i)]
            for t in range(topic_num):
                self.D_T[doc_i, t] = sum(topics == t)
        for word_i in self.id2word.keys():
            topics = topic_array[np.where(words_array == word_i)]
            for t in range(topic_num):
                self.W_T[word_i, t] = sum(topics == t)

        ## begin training
        eps = 1e-3 if topic_num > self.W_T.sum() else (topic_num / self.W_T.sum()) ** 2
        for step in range(iteration):
            pre_prob_W = self.W_T / (self.W_T.sum())
            for i, item in enumerate(topic_array):
                self.W_T[words_array[i], item] -= 1
                self.D_T[document_array[i], item] -= 1

                E_theta = (self.D_T[document_array[i], :] + alpha) / (
                self.D_T[document_array[i], :].sum() + self.D_T.shape[1] * alpha)  # E(theta(m,k))
                E_phi = (self.W_T[words_array[i], :] + beta) / (
                self.W_T[:, :].sum(axis=0) + self.W_T.shape[0] * beta)  # E(phi(k,t))
                prob = E_theta * E_phi
                topic_array[i] = np.random.choice([i for i in range(topic_num)], 1, p=prob / prob.sum())[0]
                self.D_T[document_array[i], topic_array[i]] += 1
                self.W_T[words_array[i], topic_array[i]] += 1
            # trace of the change
            prob_W = self.W_T / (self.W_T.sum())
            if np.linalg.norm(pre_prob_W - prob_W) < eps:
                print("stop step:", step)
                break
        for i, key in enumerate(self.text_dict.keys()):
            self.sample_vecs[key] = self.D_T[i, :]

    def top_N(self, asin, N):
        vec = self.sample_vecs[asin]
        return Base_Model.top_N(self, vec, N)


class SVD_Model(Base_Model):
    """
    input is tfidf matrix
    """

    def build_model(self, matrix, keys, K):
        n, p = matrix.shape
        U, Sigma, VT = np.linalg.svd(matrix)
        if K >= min(n,p):
            K = min(n,p)
        else:
            K = K
        self.vector_size = K
        svd_matrix = matrix.dot(VT[:K, :].T)*(1/Sigma)[:K]
        self.sample_vecs = dict()
        for i, key in enumerate(keys):
            self.sample_vecs[key] = svd_matrix[i, :]

    def top_N(self, asin, N):
        vec = self.sample_vecs[asin]
        return Base_Model.top_N(self, vec, N)


