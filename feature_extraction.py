# _*_ coding:utf-8 _*_

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


def dict_vector():
    # 把字典中一些类别的数据分别进行转换为特征， 原为数值的不进行转换
    # 类别特征：one-hot编码
    """
    字典数据特征抽取
    :return:None
    """
    # 字典列表数据
    dict_list = [
        {'city': '北京', 'temperature': 100},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 30}
    ]
    # 实例化
    dict_vec = DictVectorizer(sparse=False)  # sparse 是否转换为scipy.sparse矩阵表示，默认开启, 不转换是二维数组，转换后是矩阵表示
    # 调用fit_transform转换为特征抽取后的数据
    dict_data = dict_vec.fit_transform(dict_list)
    # get_feature_names()返回类别名称
    print(dict_vec.get_feature_names())
    # inverse_transform()反转特征数据
    print(dict_vec.inverse_transform(dict_data))
    print(dict_data)
    return None


def text_vector():
    """
    英文文本特征抽取
    :return: None
    """
    # 统计所有文章中所有的词，重复的统计一次
    # 在每篇文章，列表统计出现的次数
    # 单个字母不进行统计
    text_list = ["life is short,i like python", "life is too long,i dislike python"]
    text_vec = CountVectorizer()
    text_data = text_vec.fit_transform(text_list)
    print(text_data)
    print(text_vec.get_feature_names())
    # 转换为二维数组
    print(text_data.toarray())


def zh_vec():
    """
    中文文本特征抽取
    :return: None
    """
    # 定义存放待抽取特征的文章列表
    zh_list = []
    # 利用jieba分词
    con1 = jieba.cut("青年人陷于不义的时候，不敢对良心的镜子照一照；成年人却不怕正视；人生两个阶段的不同完全在于这一点。")  # 返回的是分词后的对象
    con2 = jieba.cut("一个人光溜溜地到这个世界上来，最后光溜溜地离开这个世界而去，彻底想起来，名利都是身外物，只有尽一个人的心力")
    con3 = jieba.cut("失望，有时候也是一种幸福，因为有所期待所以才会失望。因为有爱，才会有期待，所以纵使失望，也是一种幸福，虽然这种幸福有点痛。")
    con4 = jieba.cut("在这个尘世上，虽然有不少寒冷，不少黑暗，但只要人与人之间多些信任，多些关爱，那么，就会增加许多阳光。")
    # 结果转换为list
    con1_list = list(con1)
    con2_list = list(con2)
    con3_list = list(con3)
    con4_list = list(con4)
    # 转换为字符串
    c1 = ' '.join(con1_list)
    c2 = ' '.join(con2_list)
    c3 = ' '.join(con3_list)
    c4 = ' '.join(con4_list)
    zh_list.append(c1)
    zh_list.append(c2)
    zh_list.append(c3)
    zh_list.append(c4)
    # 提取特征数据
    zh_vec = CountVectorizer()
    zh_data = zh_vec.fit_transform(zh_list)
    print(zh_data)
    print(zh_vec.get_feature_names())
    print(zh_data.toarray())
    return


def tf_idf_vec():
    """
    tf: term frequency 词的频率
    idf: inverse document frequency 逆文档频率
    tf * idf :重要性程度
    对数函数: log(总文档数量/该词出现的文档数量)
    :return: None
    """
    # 定义存放待抽取特征的文章列表
    zh_list = []
    # 利用jieba分词
    con1 = jieba.cut("青年人陷于不义的时候，不敢对良心的镜子照一照；成年人却不怕正视；人生两个阶段的不同完全在于这一点。")  # 返回的是分词后的对象
    con2 = jieba.cut("一个人光溜溜地到这个世界上来，最后光溜溜地离开这个世界而去，彻底想起来，名利都是身外物，只有尽一个人的心力")
    con3 = jieba.cut("失望，有时候也是一种幸福，因为有所期待所以才会失望。因为有爱，才会有期待，所以纵使失望，也是一种幸福，虽然这种幸福有点痛。")
    con4 = jieba.cut("在这个尘世上，虽然有不少寒冷，不少黑暗，但只要人与人之间多些信任，多些关爱，那么，就会增加许多阳光。")
    # 结果转换为list
    con1_list = list(con1)
    con2_list = list(con2)
    con3_list = list(con3)
    con4_list = list(con4)
    # 转换为字符串
    c1 = ' '.join(con1_list)
    c2 = ' '.join(con2_list)
    c3 = ' '.join(con3_list)
    c4 = ' '.join(con4_list)
    zh_list.append(c1)
    zh_list.append(c2)
    zh_list.append(c3)
    zh_list.append(c4)
    # 提取特征数据
    tfidf_vec = TfidfVectorizer()
    tfidf_data = tfidf_vec.fit_transform(zh_list)
    print(tfidf_data)
    print(tfidf_vec.get_feature_names())
    print(tfidf_data.toarray())
    return None


if __name__ == '__main__':
    # 字典特征的抽取
    # dict_vector()
    # 英文特征的抽取
    # text_vector()
    # 中文特征的抽取
    # zh_vec()
    tf_idf_vec()