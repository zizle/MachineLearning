# _*_ coding:utf-8 _*_
# company: RuiDa Futures
# author: zizle
import csv
import numpy as np


def load_data():
    x = []
    y = []
    with open("Housing.csv") as f:
        reader = csv.reader(f)
        # 跳过表头
        next(reader)
        for line in reader:
            xline = [1.0]
            for s in line[:-1]:
                xline.append(float(s))
            x.append(xline)
            y.append(float(line[-1]))
    return x, y


def liner_forecast():
    x0, y0 = load_data()
    # 除了最后10行，其余转为numpy数组
    d = len(x0) - 10
    X = np.array(x0[:d])
    # y转为向量矩阵
    y = np.transpose(np.array([y0[:d]]))
    # 计算系数beta
    # 反转矩阵X
    Xt = np.transpose(X)
    XtX = np.dot(Xt, X)
    Xty = np.dot(Xt, y)
    beta = np.linalg.solve(XtX, Xty)
    # print(beta)
    # 以最后10行为原始数据，作为预测
    for data, actual in zip(x0[d:], y0[d:]):
        x = np.array([data])
        # 预测值
        prediction = np.dot(x, beta)
        prediction = "%.1f" % float(prediction[0, 0])
        print("预测值={}，实际值={}, 结果相差={}".format(prediction, float(actual), (float(prediction) - float(actual))))





if __name__ == '__main__':
    liner_forecast()
