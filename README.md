# 邮件分类系统 - 基于朴素贝叶斯

## 核心功能
本系统实现基于多项式朴素贝叶斯的邮件二分类（垃圾邮件/正常邮件），提供：
- 两种特征工程模式：高频词统计与TF-IDF加权
- 完整的文本预处理流水线
- 可配置的分类阈值调整

## 算法基础

### 多项式朴素贝叶斯
1. 条件独立性假设：
   假设每个特征（单词）在给定类别条件下相互独立：
   P(w1,w2,...,wn|c) = P(w1|c)*P(w2|c)*...*P(wn|c)

2. 贝叶斯定理应用：
   分类决策函数：
   c_hat = argmax [ logP(c) + ∑(xi * logP(wi|c)) ]
   其中：
   - P(c) 为类别先验概率
   - P(wi|c) 使用拉普拉斯平滑计算：
     (count(wi,c) + 1) / (∑count(w,c) + |V|)

## 数据处理流程

### 预处理步骤
1. 文本清洗：
   - 移除HTML标签：re.sub('<[^>]+>', '', text)
   - 去除特殊字符：保留字母数字和基本标点

2. 分词处理：
   - 英文：nltk.word_tokenize(text.lower())
   - 中文：jieba.lcut(text)

3. 停用词过滤：
   - 加载停用词表：stopwords = set(open('stopwords.txt').read().splitlines())
   - 过滤：filtered = [w for w in tokens if w not in stopwords]

4. 词形归一化：
   - 英文：PorterStemmer().stem(word)
   - 中文：保留原始词形

## 特征构建

### 高频词选择
数学表达：
特征向量 = [count(w1), count(w2), ..., count(wn)]
实现方式：
CountVectorizer(
    max_features=1000,
    min_df=3,
    binary=False
)

### TF-IDF加权
数学表达：
特征值 = tf(t,d) * log(N / (df(t) + 1))
实现方式：
TfidfVectorizer(
    ngram_range=(1,2),
    norm='l2',
    use_idf=True
)

### 关键差异
| 维度        | 高频词选择          | TF-IDF             |
|-------------|--------------------|--------------------|
| 计算复杂度  | O(n)              | O(n log n)        |
| 特征区分度  | 仅考虑频率         | 降低常见词权重     |
| 内存占用    | 较低               | 较高              |

## 特征模式切换

### 方法一：配置文件
修改config.ini：
[feature]
mode = tfidf  # 可选frequency/tfidf
max_features = 1500

### 方法二：运行时指定
初始化分类器时：
classifier = EmailClassifier(feature_mode='frequency')
或动态切换：
classifier.switch_mode('tfidf')

### 方法三：命令行参数
python train.py --feature-mode frequency
python predict.py --feature-mode tfidf

## 性能建议
- 实时系统：推荐高频词模式（响应快）
- 离线分析：推荐TF-IDF模式（准确率高）
- 内存受限：max_features建议设置1000-3000