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
    bar.add("服装",["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"],[5, 20, 36, 10, 75, 90], is_more_utils=True)

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
    title = "Profit-Loss"
    index = pd.date_range("3/8/2017", periods=6, freq="M")
    df1 = pd.DataFrame(np.random.randn(6), index=index)
    df2 = pd.DataFrame(np.random.randn(6), index=index)
    dtvalue1 = [i[0] for i in df1.values]
    dtvalue2 = [i[0] for i in df2.values]
    _index = [i for i in df1.index.format()]
    bar = Bar(title)
    bar.add("profit", _index, dtvalue1)
    bar.add("loss", _index, dtvalue2)
    bar.print_echarts_options()
    bar.render()


def setup_charts():
    """图表配置"""
    from pyecharts import Bar
    # 1. 通用配置(初始化配置)
    init_params = {
        'title': "My Fist Chart",
        'subtitle': 'sub_title',
        'width': 800,
        'height': 300,
        'title_pos': '20%'  # 标题距离左侧距离，默认'left', 有'auto', 'left','right','center'，也可以设置百分比
    }
    bar = Bar(**init_params)
    public_params = {
        'name': "服装",
        'x_axis': ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"],
        'y_axis': [5, 20, 36, 10, 75, 50],
        'is_more_utils': True,
        'is_label_show': True,  # 图形上的文本标签
        'xaxis_formatter': xaxis_formatter,
        'is_datazoom_show': False,  # 区域缩放组件，图像下方的滑动条
        'is_datazoom_extra_show': False,  # 额外的区域缩放组件，默认显示在y轴
        'is_legend_show': True,  # 图例
        'legend_pos': 'right',
        'is_visualmap': True,
        'tooltip_trigger': 'axis'


    }
    bar.add(**public_params)
    # 打印配置项
    bar.print_echarts_options()  # 打印输出图表的所有配置项
    bar.render()  # 默认在根目录下生成一个render.html文件，支持path参数


def xaxis_formatter(params):
    """x轴标签格式器，类似回调函数用法还可用在label等"""
    return "类型:"+ params


if __name__ == '__main__':
    # 开始绘图
    # bar()
    # 多次绘图
    # show_more_time()
    # 使用pandas Or numpy
    # use_pandas_or_numpy()
    # 作图配置
    setup_charts()