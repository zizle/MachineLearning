# _*_ coding:utf-8 _*_
"""
version: 0.5.11
pyecharts.Bar为柱状图
"""


def bar():
    from pyecharts import Bar
    bar = Bar("第一个图表", "副标题")
    bar.use_theme("dark")  # 0.5.2+起支持设置主题
    bar.add("服装", ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"], [5, 20, 36, 10, 75, 90], is_more_utils=True)
    # 打印配置项
    bar.print_echarts_options()  # 打印输出图表的所有配置项
    bar.render()  # 默认在根目录下生成一个render.html文件，支持path参数


def show_more_time():
    """多次显示图表"""
    from pyecharts import Bar, Line
    from pyecharts.engine import create_default_environment

    bar = Bar("我的第一个图表", "副标题")
    bar.add("服装", ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"], [5, 20, 36, 10, 75, 90], is_more_utils=True)

    line = Line()
    line.add("服装", ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"], [5, 20, 36, 10, 75, 90], is_more_utils=True)

    # create_default_environment(fileType) 为默认图表创建一个默认配置环境，fileType:'html', 'svg', 'png', 'jpeg', 'gif', 'pdf'
    env = create_default_environment("html")
    env.render_chart_to_file(bar, path='bar.html')
    env.render_chart_to_file(line, path='line.html')


def use_pandas_or_numpy():
    """使用numpy整数类型要确保为int，而不是numpy.int32"""
    import pandas as pd
    import numpy as np
    from pyecharts import Bar
    title = "示例-柱状图"
    index = pd.date_range("3/8/2017", periods=6, freq="M")
    df1 = pd.DataFrame(np.random.randn(6), index=index)
    df2 = pd.DataFrame(np.random.randn(6), index=index)
    dtvalue1 = [i[0] for i in df1.values]
    dtvalue2 = [i[0] for i in df2.values]
    _index = [i for i in df1.index.format()]

    # attr = ["{}月".format(i) for i in range(1, 13)]
    # v1 = [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.6, 32.6, 20.0, 6.4, 3.2]
    # v2 = [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.2, 186.3, 48.6, 18.9, 6.0, 2.1]
    bar = Bar(title)
    bar.add("profit", _index, dtvalue1)
    bar.add("loss", _index, dtvalue2)
    bar.render()


if __name__ == '__main__':
    # 开始绘图
    # bar()
    # 多次绘图
    # show_more_time()
    # 使用pandas Or numpy
    use_pandas_or_numpy()