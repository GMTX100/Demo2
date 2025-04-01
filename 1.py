import re
import numpy as np
from jieba import cut
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            line = cut(line)
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words

# 读取训练数据并处理成TF-IDF特征
train_files = ['邮件_files/{}.txt'.format(i) for i in range(151)]
train_texts = [' '.join(get_words(f)) for f in train_files]

# 初始化TF-IDF向量化器，限制特征数为100
vectorizer = TfidfVectorizer(max_features=100, tokenizer=lambda x: x.split(), token_pattern=None)
X_train = vectorizer.fit_transform(train_texts).toarray()

labels = np.array([1] * 127 + [0] * 24)

# 训练模型
model = MultinomialNB()
model.fit(X_train, labels)

def predict(filename):
    """对未知邮件分类"""
    # 处理文本并转换为TF-IDF特征
    words = get_words(filename)
    text = ' '.join(words)
    X_test = vectorizer.transform([text]).toarray()
    result = model.predict(X_test)
    return '垃圾邮件' if result == 1 else '普通邮件'

# 测试分类结果
print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt')))
print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt')))
print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt')))
print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt')))
print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt')))