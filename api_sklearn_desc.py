# _*_ coding:utf-8 _*_
# company: RuiDa Futures
# author: zizle


""" 
1. 内置数据集 
    如：鸢尾花数据集 - from sklearn.datasets import load_iris  --> iris = load_iris()
    详见：datasets_iris()函数

2. 特征提取 - from sklearn.feature_extraction

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
    详见： textToVectorEnglish()、textToVectorChinese()和textToVectorAutoChinese()函数


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


if __name__ == '__main__':
    textToVectorAutoChinese()

