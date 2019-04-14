import itertools
import time
import os
import re
from collections import Counter
from itertools import repeat
from functools import reduce
from multiprocessing import Pool

import jieba
from hanziconv import HanziConv

dataset_path = '/Users/luoqi/projects/AI-for_NLP/datasets/AA'
pool_size = 3
voc_size = 2000


def preprocess(file, debug='no'):
    """read and clean file """
    with open(file) as f:
        content = f.read()
        if debug == 'yes':
            content = content[:200]
        #print(len(content))

        # romove tags
        content = re.sub(r'<\/?doc.*>', '', content)

        # to simplified Chinese
        content = HanziConv.toSimplified(content)

        # find sentences
        sentences = re.split('。', content)
        # print('original sentences:\n', sentences)

        # keep words and digits only
        sentences = [''.join(re.findall('[\w|\d]+', sen)) for sen in sentences if sen.strip() and len(sen.strip()) > 0]
        # print('\ncleaned sentences\n', sentences)

        # tokenize
        tokenized_sentences = [list(jieba.cut(sen)) for sen in sentences if sen.strip() and len(sen.strip()) > 0]
        # print('\ntokenized sentences\n', tokenized_sentences)

        # add <s> and </s> into each sentences, indicating start and end of a sentence
        for sen in tokenized_sentences:
            sen.append('</s>')
            sen.insert(0, '<s>')
        # print('\nafter add <s> and </s>:\n', tokenized_sentences)

        # join to one list
        corpus = list(itertools.chain.from_iterable(tokenized_sentences))

        return corpus


def get_all_files(dataset_path):
    """get all file paths from specific dataset path"""
    all_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def get_corpus(dataset_path, debug='no'):
    assert os.path.isdir(dataset_path), 'invalid dataset path.'
    all_files = get_all_files(dataset_path)
    corpus = []
    if len(all_files) > 0:
        with Pool(pool_size) as p:
            tokens_list = p.starmap(preprocess, zip(all_files, repeat(debug)))
        corpus = list(itertools.chain.from_iterable(tokens_list))

    # get valid tokens according to volcabulary size
    token_cnt = Counter(corpus)
    valid_tokens = token_cnt.most_common(voc_size)
    drop_tokens = []
    for token, cnt in token_cnt.items():
        if token not in valid_tokens:
            drop_tokens.append(token)

    for i, t in enumerate(corpus):
        if corpus[i] in drop_tokens:
            corpus[i] = 'UKN'
    return corpus


def product(nums):
    return reduce(lambda x1, x2: x1*x2, nums)


def get_prob(token, token_cnt):
    cnt_sum = sum(token_cnt.values())
    eps = 1 / cnt_sum
    if token in token_cnt:
        return token_cnt[token] / cnt_sum
    else:
        return eps


def unigram_lm(string, corpus):
    token_cnt = Counter(corpus)
    return product(get_prob(t, token_cnt) for t in jieba.cut(string))


def get_combination_prob(w1, w2, _2_gram_counter, _2_gram_sum):
    if w1 + w2 in _2_gram_counter: return _2_gram_counter[w1+w2] / _2_gram_sum
    else:
        return 1 / _2_gram_sum


def get_prob_2_gram(w1, w2, token_cnt, _2_gram_counter, _2_gram_sum):
    return get_combination_prob(w1, w2, _2_gram_counter, _2_gram_sum) / get_prob(w1, token_cnt)


def two_gram_lm(string, corpus):
    token_cnt = Counter(corpus)
    all_2_grams_words = [''.join(corpus[i:i + 2]) for i in range(len(corpus[:-2]))]
    _2_gram_sum = len(all_2_grams_words)
    _2_gram_counter = Counter(all_2_grams_words)

    words = jieba.cut(string)

    sentence_probability = 1
    for i, word in enumerate(words):
        if i == 0:
            prob = get_prob(word, token_cnt)
        else:
            previous = words[i - 1]
            prob = get_prob_2_gram(previous, word, token_cnt, _2_gram_counter, _2_gram_sum)
        sentence_probability *= prob
    return sentence_probability


if __name__ == '__main__':
    t1 = time.time()
    corpus = get_corpus(dataset_path, debug='yes')
    t2 = time.time()
    print(f'it took {t2 - t1:.2f}s to get corpus. ', end='')
    print('length of corpus is:',len(corpus))

    s1 = "数学蕴含无穷力量"
    s2 = "我喜欢数学"

    prob2 = unigram_lm(s2, corpus)
    print('prob of s1:', prob2)

