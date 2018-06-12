# _*_ coding:utf-8 _*_

# 把字典中一些类别的数据分别进行转换为特征， 原为数值的不进行转换
# 类别特征：one-hot编码

from sklearn.feature_extraction import DictVectorizer


def dict_vector():
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
    data = dict_vec.fit_transform(dict_list)
    # get_feature_names()返回类别名称
    print(dict_vec.get_feature_names())
    print(data)
    return None


if __name__ == '__main__':
    dict_vector()