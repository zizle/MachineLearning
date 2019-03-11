# _*_ coding:utf-8 _*_
# company: RuiDa Futures
# author: zizle


""" 
1. 内置数据集 
    如：鸢尾花数据集 - from sklearn.datasets import load_iris  --> iris = load_iris()
    详见：datasets_iris()函数

2. 特征工程：
  2.1 特征提取 - from sklearn.feature_extraction
    如：字典特征提取(类别 -> one-hot编码) - sklearn.feature_extraction.DictVectorizer(sparse=True), DictVectorizer.fit_transform(data)
        -> 实例化时sparse=True返回的是一个sparse矩阵(稀疏矩阵)  -> (行, 列)  数值  -> 转为二维数组使用方法toarray()
        -> 实例化时sparse=False返回的是一个二维数组
    详见：dictToVector()函数
    应用场景：
        1. 将数据集特征转换为字典类型
        2. 本身数据就是字典类型

    如：
    文本特征提取(单词作为特征) - (1).sklearn.feature_extraction.text.CountVectorizer(stop_words=[]), 返回sparse矩阵
    参数stop_words为停用词，有现成停用词表
    (2).sklearn.feature_extraction.text.Tfidfvectorizer()利用此的逆向文档数据(一个词的普遍重要程度),衡量词的重要性程度。
    详见： textToVectorEnglish()、textToVectorChinese()和textToVectorAutoChinese()函数
          textToVectorChineseTfidf()

  2.1 特征预处理：通过一些转换函数将特征数据转换为更加适合算法模型的特征数据的过程，数据无量纲化
    为什么要做归一化和标准化，特征的单位大小相差太大，会影响学习结果
    归一化计算方式： 缺点（鲁棒性较差，易受数据异常值影响，适合精确的数据场景）
    90  2   10  40
    60  4   15  45
    75  3   13  46
    X' = (x - min) / (max - min)  # (90 - 60) / (90 - 60) = 1
    X'' = X' * (mx - mi) + mi  # mx和mi为想要方法的区间数，如0~1，mx=1， mi=0  # 1 * (1 - 0) + 0 = 1

    标准化计算方式
    将原始的数据进行变换，变为均值为0，标准差为1的范围内
    X' = (x - avg) / (sigma)  # sigma为标准差  标准差衡量的是离散程度

    详见函数：
        minMaxDealData()、standScalarDealData()函数

  2.2 特征降维
    降低特征的个数，得到特征与特征之间是不相关的
    特征选择方法：1 filter过滤式（方差选择法 - 低方差特征过滤、相关系数 - 特征与特征之间的相关性）
               2 embeded嵌入式（决策树、正则化、深度学习）

    API
    # 低方差特征过滤  sklearn.feature_selection.VarianceThreshold(threshold = 0.0)  # 参数为设置的方差，删掉比其小的特征




"""


def datasets_iris():
    """内置鸢尾花数据集"""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    # print("鸢尾花数据集:\n", iris)
    # 数据集的划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train)
    print("训练集的大小：\n", x_train.shape)
    print("测试集的特征值：\n", x_test)
    print("测试集的大小：\n", x_test.shape)
    print("训练集的目标值：\n", y_train)
    print("测试集的目标值：\n", y_test)


def dictToVector():
    from sklearn.feature_extraction import DictVectorizer
    """
    字典特征提取
    类别-> one-hot编码
    """
    data = [
        {"city": "北京", "temperature": 100},
        {"city": "上海", "temperature": 60},
        {"city": "深圳", "temperature": 30}
    ]
    transfer = DictVectorizer(sparse=False)
    data_new = transfer.fit_transform(data)
    feature_names = transfer.get_feature_names()
    print("特征提取结果:\n", data_new)
    print("特征名字:\n", feature_names)


def textToVectorEnglish():
    """
    文本特征提取
    统计特征词出现的次数
    """
    from sklearn.feature_extraction.text import CountVectorizer
    data = [
        "Life is short, i like python python",
        "Life is too long, i dislike python"
    ]
    transfer = CountVectorizer(stop_words=['is', 'too'])
    data_new = transfer.fit_transform(data)
    feature_names = transfer.get_feature_names()
    print("特征提取结果:\n", data_new.toarray())
    print("特征名字:\n", feature_names)


def textToVectorChinese():
    """
    文本特征提取
    统计特征词出现的次数
    """
    from sklearn.feature_extraction.text import CountVectorizer
    # 手动分词
    data = [
        "我 爱 北京 天安门",
        "天安门 上 太阳 升"
    ]
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data)
    feature_names = transfer.get_feature_names()
    print("特征提取结果:\n", data_new.toarray())
    print("特征名字:\n", feature_names)


def textToVectorAutoChinese():
    """
    文本特征提取,自动分词
    统计特征词出现的次数
    """
    import jieba
    from sklearn.feature_extraction.text import CountVectorizer
    # 中文提取需要使用分词
    data = [
        "2018年，我国经济总量突破90万亿元，经济运行稳中有进，转型升级蹄疾步稳。",
        "全国居民人均可支配收入达28228元，民生福祉不断改善，人民获得感明显增强。",
        "今天的节目，就让我们走进重庆的一家基层卫生院，看看那里发生的故事。"
    ]
    split_data = []
    # 遍历句子分词
    for sent in data:
        split_data.append(" ".join(list(jieba.cut(sent))))

    transfer = CountVectorizer()
    data_new = transfer.fit_transform(split_data)
    feature_names = transfer.get_feature_names()
    print("特征提取结果:\n", data_new.toarray())
    print("特征名字:\n", feature_names)


def textToVectorChineseTfidf():
    """利用TF-IDF方法进行文本特征抽取"""
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    # 中文提取需要使用分词
    data = [
        "2018年，我国经济总量突破90万亿元，经济运行稳中有进，转型升级蹄疾步稳。",
        "全国居民人均可支配收入达28228元，民生福祉不断改善，人民获得感明显增强。",
        "今天的节目，就让我们走进重庆的一家基层卫生院，看看那里发生的故事。"
    ]
    split_data = []
    # 遍历句子分词
    for sent in data:
        split_data.append(" ".join(list(jieba.cut(sent))))

    transfer = TfidfVectorizer()
    data_new = transfer.fit_transform(split_data)
    feature_names = transfer.get_feature_names()
    print("特征提取结果:\n", data_new.toarray())
    print("特征名字:\n", feature_names)


def minMaxDealData():
    """数据归一化处理"""
    from sklearn.preprocessing import MinMaxScaler
    # 准备数据
    import numpy as np
    data = np.array([
        [40920, 8.32, 0.9539],
        [14488, 7.15, 1.6739],
        [26052, 1.44, 0.8051],
        [75136, 13.14, 0.4268],
        [38344, 1.66, 0.1342]
    ])
    # 实例化转换器类
    transfer = MinMaxScaler(feature_range=(3, 4))  # 参数为目标数据范围
    data_new = transfer.fit_transform(data)
    print(data_new)


def standScalarDealData():
    """标准化"""
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    data = np.array([
        [40920, 8.32, 0.9539],
        [14488, 7.15, 1.6739],
        [26052, 1.44, 0.8051],
        [75136, 13.14, 0.4268],
        [38344, 1.66, 0.1342]
    ])
    # 实例化转换器类
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)



if __name__ == '__main__':
    # 内置鸢尾花数据集
    # datasets_iris()
    # 字典特征提取
    # dictToVector()
    # 英语文本特征提取
    # textToVectorEnglish()
    # 中文文本提取
    # textToVectorChinese()
    # 中文文本自动分词特征提取
    # textToVectorAutoChinese()
    # 词的重要性程度抽取
    # textToVectorChineseTfidf()
    # 归一化
    # minMaxDealData()
    # 标准化
    standScalarDealData()
