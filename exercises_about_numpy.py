# _*_ coding:utf-8 _*_
# company: RuiDa Futures
# author: zizle
# date: 2019-02-28
import numpy as np


def one():
    print(np.__version__)


def two():
    print(np.arange(10))


def three():
    a = np.full((2, 2), True, dtype=bool)
    print(a)

    b = np.ones((3, 3), dtype=bool)
    print(b)


def four():
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(arr[arr % 2 == 1])


def five():
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    arr[arr % 2 == 0] = 0
    print(arr)


def six():
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    out = np.where(arr % 2 == 1, -1, arr)
    print(out)


def seven():
    arr = np.arange(10)
    arr = arr.reshape((2, -1))  # 参数为(行数,列数)，列数为-1时，自动分列数
    print(arr)


def eight():
    a = np.arange(10).reshape(2, -1)
    b = np.repeat(1, 10).reshape(2, -1)
    # 1
    # out = np.concatenate([a, b], axis=0)
    # 2
    # out = np.vstack([a, b])
    # 3
    out = np.r_[a, b]
    print(out)


def nine():
    a = np.arange(10).reshape(2, -1)
    b = np.repeat(1, 10).reshape(2, -1)
    # 1
    # out = np.concatenate([a, b], axis=1)
    # 2
    # out = np.hstack([a, b])
    # 3
    out = np.c_[a, b]
    print(out)


def ten():
    a = np.array([1, 2, 3])
    out1 = np.repeat(a, 3)
    out2 = np.tile(a, 3)
    print(out1)
    print(out2)
    print(np.r_[out1, out2])


if __name__ == '__main__':
    """
    1、导入numpy作为np，并查看版本
    难度等级：L1
    问题：将numpy导入为 np 并打印版本号。
    """
    # one()
    """
    2、如何创建一维数组？
    难度等级：L1
    问题：创建从0到9的一维数字数组
    期望输出：
    # > array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """
    # two()
    """
    3. 如何创建一个布尔数组？
    难度等级：L1
    问题：创建一个numpy数组元素值全为True（真）的数组
    """
    # three()
    """
    4. 如何从一维数组中提取满足指定条件的元素？
    难度等级：L1
    问题：从 arr 中提取所有的奇数
    给定：
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    期望的输出：
    # > array([1, 3, 5, 7, 9])
    """
    # four()
    """
    5. 如何用numpy数组中的另一个值替换满足条件的元素项？
    难度等级：L1
    问题：将arr中的所有奇数替换为-1。
    给定：
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    期望的输出：
    # >  array([0, 1, 0, 3, 0, 5, 0, 7, 0, 9)
    """
    # five()
    """
    6. 如何在不影响原始数组的情况下替换满足条件的元素项？
    难度等级：L2
    问题：将arr中的所有奇数替换为-1，而不改变arr。
    给定：
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    期望的输出：
    out
    # >  array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])
    arr
    # >  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """
    # six()
    """
    7. 如何改变数组的形状？
    难度等级：L1
    问题：将一维数组转换为2行的2维数组
    给定：
    np.arange(10)
    # > array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    期望的输出：
    # > array([[0, 1, 2, 3, 4],
    # >        [5, 6, 7, 8, 9]])
    """
    # seven()
    """
    8. 如何垂直叠加两个数组？
    难度等级：L2
    问题：垂直堆叠数组a和数组b
    给定：
    a = np.arange(10).reshape(2,-1)
    b = np.repeat(1, 10).reshape(2,-1)
    期望的输出：
    # > array([[0, 1, 2, 3, 4],
    # >        [5, 6, 7, 8, 9],
    # >        [1, 1, 1, 1, 1],
    # >        [1, 1, 1, 1, 1]])
    """
    # eight()
    """
    9. 如何水平叠加两个数组？
    难度等级：L2 
    问题：将数组a和数组b水平堆叠。
    给定：
    a = np.arange(10).reshape(2,-1)
    b = np.repeat(1, 10).reshape(2,-1)
    期望的输出：
    # > array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
    # >        [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
    """
    # nine()
    """
    10. 如何在无硬编码的情况下生成numpy中的自定义序列？
    难度等级：L2
    问题：创建以下模式而不使用硬编码。只使用numpy函数和下面的输入数组a。
    给定：
    a = np.array([1,2,3])
    期望的输出：
    # > array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    """
    ten()
