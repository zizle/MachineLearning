# _*_ coding:utf-8 _*_

# 特征预处理：通过特定的统计方法将数据转换为算法要求的数据：1、归一化 2、标准化 3. (特征的)降维:特征选择；主成分分析
# 数值型数据：标准缩放(1、归一化 2、标准化 3、缺失值)
# 类别型数据： one-hot编码
# 时间类型： 时间的切分
# sklearn 归一化： sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)...)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


def min_max():
    """
    归一化处理
    目的： 使某一个特征对最终结果不会造成更大的影响(统计者认为每个特征同等重要)
    缺点： 对异常点处理不好，所以这方法鲁棒性差，只适合传统精确的小数据场景
    公式：
    X1​​ = ​(x−min)/(max−min);
    X2 = X1 * (mx-mi) + mi
    :return:None
    """
    # 准备数据： 针对二维数组
    list_array = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]
    # 实例化
    mm = MinMaxScaler(feature_range=(0, 1))
    mm_data = mm.fit_transform(list_array)
    print(mm_data)


def standard():
    """
    标准化： 使得某一个特征对最终结果不会造成更大的影响
    特点： 把数据变化为均值为0， 标准差为1范围内
    当有一定的数据量后，少量的异常点对结果的影响不大
    公式：
        X1 = (x - 平均值)/ 标准差
        标准差 = 方差开方
        方差 = ((x1-平均值)平方 + (x2-平均值)平方 + ...) / 每个特征样本数
        方差是考量数据的稳定性
    # sclearn 标准化： sklearn.preprocessing.StandardScaler()
    平均值：StandardScaler.mean_
    方差：StandardScaler.std_   # 测试，不可用
    :return:
    """
    list_array = [[1, -1, 3], [2, 4, 2], [4, 6, -1]]
    std = StandardScaler()
    std_data = std.fit_transform(list_array)
    print(std_data)


# 特征选择：三大神器：1.过滤式(Filter:VarianceThreadhold)2嵌入式(Embedded:正则化，决策树)3 包裹式(Wrapper)
def delete_var():
    """
    删除低方差的特征: 去掉差不多的数据
    :return: None
    """
    lists = [[0, 2, 0, 3],
             [0, 1, 4, 3],
             [0, 1, 1, 3]
             ]

    var = VarianceThreshold(threshold=0.0)

    var_data = var.fit_transform(lists)

    print(var_data)
    return None


def main_composition():
    """
    主成分分析：PCA，尽可能降低数据的维度，损失少量的信息， 可以削减回归分析或者聚类分析中特征的数量
    :return: None
    """
    lists = [
        [2, 8, 4, 5],
        [6, 3, 0, 8],
        [5, 4, 9, 1]
    ]
    # n_components: 小数 0.9~0.95；整数：减少到的特征数量
    pca = PCA(n_components=0.95)

    pca_data = pca.fit_transform(lists)

    print(pca_data)


if __name__ == '__main__':
    # 归一化
    # min_max()
    # 标准化
    # standard()
    # 降维
    # delete_var()
    # 主成分分析
    main_composition()
