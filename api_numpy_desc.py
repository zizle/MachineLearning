# _*_ coding:utf-8 _*_
# company: RuiDa Futures
# author: zizle
import numpy as np
import matplotlib.pyplot as plt

"""
1. 数组的处理与计算

2. Numpy中的数组
    详见：arrayInNumpy()函数
    
3. Numpy中的数组操作
    详见: operationNumpyArray()函数


"""


def arrayInNumpy():
    """
    Numpy中的数组
    :return: None
    """
    """一维数组"""
    my_array = np.array([1, 2, 3, 4, 5])
    print(my_array.shape)
    # 包含5个0的数组
    zeros_array = np.zeros(5)
    ones_array = np.ones(5)
    random_array = np.random.random(6)
    print(zeros_array, ones_array, random_array)
    """二维数组"""
    my_2d_zeros_array = np.zeros((3, 3))
    print("二维数组：\n", my_2d_zeros_array)
    """数组的取值"""
    get_array_num = np.array([[1, 2, 4, 7], [3, 5, 6, 9], [4, 7, 3, 8]])
    print("获取数组的第2行第2列的数字为:", get_array_num[1][1])
    print("获取数组全部第3列的值为：", get_array_num[:, 2])  # 行号为：， 列号为索引
    print("获取数组全部第2行的值为：", get_array_num[1, :])  # 列号为索引，行号为：


def operationNumpyArray():
    """操作Numpy数组"""
    a = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]])
    b = np.array([[5.0, 6.0, 7.0], [7.0, 8.0, 9.0], [3.0, 4.0, 5.0]])
    print("数组1:\n", a)
    print("数组2:\n", b)
    print("求和结果:\n", a + b)
    print("求差结果:\n", a - b)
    print("求积结果:\n", a * b)  # 执行的是元素乘法
    print("求商结果:\n", a / b)
    print("求次方结果:\n", a ** 2)

    print("矩阵乘法结果:\n", a.dot(b))

    """多维数组切片 -- 传入列表为参数，0索引为行的切片，1索引为列的切片"""
    mdarray = np.array([[11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                        [26, 27, 28, 29, 30],
                        [31, 32, 33, 34, 35]])
    print("第1行，第2,3,4切出:\n", mdarray[0, 1:4])
    print("切出前3行第2列:\n", mdarray[0:3, 1])
    print("切出第2-4行的3-5列:\n", mdarray[1:4, 2:5])
    new_array = mdarray[1:4, 2:5]
    new_array[0, 0] = 88  # 特别注意！改变切片出来的数组将直接改变原数组
    print(mdarray)

    """数组的属性"""
    print("类型:", type(mdarray))
    print("数据类型", mdarray.dtype)
    print("数组样式", mdarray.shape)
    print("每项占用字节数", mdarray.itemsize)
    print("数组的维数", mdarray.ndim)
    print("所有数据消耗的字节数", mdarray.nbytes)

    """数组运算符"""
    c_array = np.arange(10)
    print(c_array)
    print(c_array.sum())  # >>>45
    print(c_array.min())  # >>>0
    print(c_array.max())  # >>>9
    print(c_array.cumsum())  # >>>[ 0  1  3  6 10 15 21 28 36 45]

    """花式索引 -- 获取数组中我们想要的特定元素"""
    d_array = np.arange(0, 100, 10)
    print(d_array)
    index = [1, 5, -1]
    print(d_array[index])

    """布尔屏蔽 -- 根据我们指定的条件检索数组中的元素"""
    e_array = np.linspace(0, 2 * np.pi, 50)  # 返回指定间隔等距数据
    print(e_array)
    f_array = np.sin(e_array)
    print(f_array)
    plt.plot(e_array, f_array)
    mask = f_array >= 0
    plt.plot(e_array[mask], f_array[mask], 'ro')
    mask = (f_array >= 0) & (e_array <= np.pi / 2)
    plt.plot(e_array[mask], f_array[mask], 'bo')
    plt.show()

    """缺省索引 -- 从多维数组的第一个维度获取索引或切片的一种方便方法"""
    g_array = np.arange(0, 100, 10)
    h_array = g_array[:5]
    i_array = g_array[g_array >= 50]
    print(g_array)
    print(h_array)
    print(i_array)

    """where()函数 -- 根据条件返回数组中的索引"""
    j_array = np.where(g_array < 50)
    k_array = np.where(g_array >= 50)[0]
    print(j_array)
    print(k_array)


def slicing_numpy_array():
    """索引与切片混用"""
    a = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ])
    row_r1 = a[1, :]
    row_r2 = a[1:2, :]
    print(row_r1)
    print(row_r2)
    col_r1 = a[:, 1]
    col_r2 = a[:, 1:2]
    print(col_r1)
    print(col_r2)

    """整数数组索引"""
    print(a[[0, 1, 2], [1, 0, 1]])  # 整数数组索引定义行列关系，数组第一行为行索引，第二行为列索引，即
    # [a[0, 0], a[1, 1], a[2, 0]]
    # 整数数组索引的一个有用技巧是从矩阵的每一行中选择或改变一个元素：
    b = np.array([0, 2, 0])
    print(np.arange(3), b)
    print("-" * 50)
    print(a[np.arange(3), b])
    a[np.arange(3), b] += 10
    print(a)

    """布尔索引"""
    bool_idx = (a > 10)
    print(bool_idx)
    print(a[a > 10])


def data_type():
    """Numpy在创建数组时尝试猜测数据类型，但构造数组的函数通常还包含一个可选参数来显式指定数据类型"""
    x = np.array([1, 2])  # Let numpy choose the datatype
    print(x.dtype)  # Prints "int64"

    x = np.array([1.0, 2.0])  # Let numpy choose the datatype
    print(x.dtype)  # Prints "float64"

    x = np.array([1, 2], dtype=np.int64)  # Force a particular datatype
    print(x.dtype)  # Prints "int64"


def math_in_array():
    """数组中的数学"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    # 相加
    print("x+y: \n", x + y)
    print("np.add(x,y):\n", np.add(x, y))
    # 相减
    print(x - y)
    print(np.subtract(x, y))
    # 相乘
    print(x * y)
    print(np.multiply(x, y))
    # 相除
    print(x / y)
    print(np.divide(x, y))
    # 开方
    print(np.sqrt(x))

    """我们使用dot函数来计算向量的内积，将向量乘以矩阵, dot既可以作为numpy模块中的函数，也可以作为数组对象的实例方法"""
    v = np.array([9, 10])
    m = np.array([11, 12])
    print(v.dot(x))
    print(np.dot(v, x))

    """numpy之sum函数   完整的数学函数：https://www.numpy.org.cn/reference/routines/math.html"""
    print(np.sum(x))  # 将数组内的数求和
    print("sum(x, axis=0)\n", np.sum(x, axis=0))  # 纵向求和
    print("sum(x, axis=1)\n", np.sum(x, axis=1))  # 横向求和


def boarding():
    """广播（特性） - 允许numpy在执行算术运算时使用不同形状的数组。当有较小数组和较大数组，并使用较小数组来对较大数组进行操作"""
    # 矩阵x
    x = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    # 向量v
    v = np.array([1, 0, 1])
    y = np.empty_like(x)
    # 手动赋值元素
    for i in range(4):
        y[i, :] = x[i, :] + v

    # 拆分步骤，我们可以理解在矩阵x右边堆叠相应行数的v，再执行矩阵相加
    vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
    z = x + vv
    """
    numpy广播是我们不用创建这个堆叠的矩阵vv：
    将两个数组一起广播遵循以下规则：
        1 如果数组不具有相同的rank，则将较低等级数组的形状添加1，直到两个形状具有相同的长度。
        2 如果两个数组在维度上具有相同的大小，或者如果其中一个数组在该维度中的大小为1，则称这两个数组在维度上是兼容的。
        3 如果数组在所有维度上兼容，则可以一起广播。
        4 广播之后，每个阵列的行为就好像它的形状等于两个输入数组的形状的元素最大值。
        5 在一个数组的大小为1且另一个数组的大小大于1的任何维度中，第一个数组的行为就像沿着该维度复制一样
    """
    ay = x + v
    # print(ay)  # 结果与创建堆叠矩阵再相加是一致的
    # 1-广播的应用：计算向量积
    v = np.array([1, 2, 3])
    w = np.array([4, 5])
    # 整形
    t = np.reshape(v, (3, 1))
    # print(t)
    # print(t * w)  # 对w进行广播
    # 2-广播的应用：矩阵每行每个元素添加向量
    # print(x + v)
    # 3-广播的应用：矩阵每列每个元素添加向量
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print((x.T + w).T)  # 旋转矩阵，添加向量，转回原样
    print(np.reshape(w, (2, 1)) + x)
    # 4-广播的应用：矩阵乘以常数
    print(x * 2)


def scipy_handle_img():
    """用Scipy处理图像 需安装PIL库"""
    from scipy.misc import imread, imresize, imsave
    img = imread("data/img/cat.jpg")
    print(img.dtype, img.shape)
    print(img.size)
    print("============")
    # tinted = img + [100, 50, 70]
    tinted = imresize(img, (650, 300))
    imsave("data/img/tinted_cat.jpg", tinted)


def distance():
    """scipy.spatial.distance.pdist算给定集合中所有点对之间的距离："""
    from scipy.spatial.distance import pdist, squareform
    x = np.array([[0, 1], [1, 0], [2, 0]])
    print(x)

    d = squareform(pdist(x, 'euclidean'))  # euclidean 欧几里得距离（在m维空间里两个点真实距离）
    print(d)


def create_array():
    # 创建一维数组
    a = np.arange(20)  # 与python列表不同的是numpy数组元素是同质的
    print(a, type(list(a)))
    # 创建二维数组
    b = np.arange(20).reshape(4, 5)
    print(b)
    # 创建三维数组及更多维度
    c = np.arange(27).reshape(3, 3, 3)
    print(c)
    # 使用arange函数，你可以创建一个在定义的起始值和结束值之间具有特定序列的数组。
    d = np.arange(10, 35, 3)
    print(d)
    # 使用其他numpy函数
    e = np.zeros((2, 4))
    print(e)
    f = np.ones((3, 4))
    print(f)
    # empty函数创建一个数组。它的初始内容是随机的，取决于内存的状态。
    print('==============')
    g = np.empty((2, 3))
    print(g)
    # full函数创建一个填充给定值的n * n数组。
    h = np.full((2, 3), 5)
    print(h)
    # eye函数可以创建一个n * n矩阵，对角线为1s，其他为0。
    i = np.eye(3, 3)
    print(i)
    # 函数linspace在指定的时间间隔内返回均匀间隔的数字。 例如，下面的函数返回0到10之间的四个等间距数字。
    j = np.linspace(0, 20, num=5)
    print(j)
    # Python列表转换
    k = np.array([])  # 参数为列表
    print(k, type(k))
    # 使用特殊的库函数
    # 创建元素为0-1之间的random函数
    l = np.random.random((3, 4))
    print(l)
    # 结语：创建和填充Numpy数组是使用Numpy执行快速数值数组计算的第一步。


def matrix_vector():
    # 构造矩阵只需吧相应的行作为列表传入即可
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    print(matrix)
    # 构造向量我们会这样
    vector_a = np.array([[1], [2], [3]])
    print(vector_a)
    # 转置行向量
    vector_b = np.transpose(np.array([[1, 2, 3]]))
    print(vector_b)
    # 重载数组索引和切片符号访问矩阵的各个部分
    print(matrix[1, 2])
    # 切出第二列
    print(matrix[:, 0:1])  # matrix[行索引,列索引]  规则：左闭右开
    # 乘法计算
    w = np.dot(matrix, vector_a)
    print(w)
    # 用numpy求解方程组 - 线性代数求解矩阵向量方程 Ax = b , 求向量x
    A = np.array([[2, 1, -2], [3, 0, 1], [1, 1, -1]])
    b = np.transpose(np.array([[-3, 5, -2]]))
    x = np.linalg.solve(A, b)
    print(x)
    # 求证
    print(np.dot(A, x))
    # 实例见包“linear”


if __name__ == '__main__':
    print(np.__version__)  # 查看numpy版本
    # Numpy中的数组
    # arrayInNumpy()
    # Numpy中的数组操作
    # operationNumpyArray()
    # 数组切片的复习
    # slicing_numpy_array()
    # 数组的数据类型
    # data_type()
    # 数组中的数学
    # math_in_array()
    # 广播
    # boarding()
    """Scipy的使用"""
    # 1 scipy处理图像
    # scipy_handle_img()
    # 2 函数 scipy.io.loadmat 和 scipy.io.savemat 允许你读取和写入MATLAB文件。你可以在这篇文档中学习相关操作。
    # 3  SciPy定义了一些用于计算点集之间距离的有用函数。
    # distance()
    """numpy创建数组的方式"""
    # create_array()
    """numpy中的矩阵和向量"""
    # matrix_vector()

