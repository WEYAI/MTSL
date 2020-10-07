import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


# 添加数据标签 就是矩形上面的数值
# def add_labels(rects):
#     for rect in rects:
#         height = rect.get_height()
#         plt.text(rect.get_x() + rect.get_width() / 2, height + 0.01 * height, '%.2f' % height, ha='center',
#                  va='bottom', fontsize=10, color='black')
#         rect.set_edgecolor('white')


if __name__ == '__main__':
    # 这两行代码解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 输入统计数据
    task = (
        'CARDINAL', 'DATE', 'FAC', 'EVENT', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG',
        'PERCENT',
        'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART')
    baseline_F1 = [83.35, 83.06, 82.99, 83.06, 83.35, 83.06, 82.99, 83.06, 83.35, 83.06, 82.99, 83.06, 83.35, 83.06,
                   82.99, 83.06, 83.35, 84.7]
    ours_F1 = [83.35, 83.06, 82.99, 83.06, 83.35, 83.06, 82.99, 83.06, 83.35, 83.06, 82.99, 83.06, 83.35, 83.06, 82.99,
               83.06, 83.35, 84.7]
    # ValueError: shape mismatch: objects cannot be broadcast to a single shape
    # 可能是条形图的x和y的数组长度不同造成的，需要修改数据，保持一致
    bar_width = 0.3  # 条形宽度
    baseline = np.arange(len(task))  # baseline条形图的横坐标
    our_model = baseline + bar_width  # 自己的模型的条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    baseline_figure = plt.bar(baseline, height=baseline_F1, width=bar_width, color='grey', label='Thai-Hoang Pham')
    our_model_figure = plt.bar(our_model, height=ours_F1, width=bar_width, color='black', label='MTSL-RN')
    # add_labels(baseline_figure)
    # add_labels(our_model_figure)
    plt.ylim(0, 100)  # set the y alias
    plt.legend()  # 显示图例
    plt.xticks(baseline + bar_width / 2, task)  # 让横坐标轴刻度显示 task 的F1值， baseline + bar_width/2 为横坐标轴刻度的位置

    # plt.yticks()
    plt.ylabel('调和平均数')  # 纵坐标轴标题
    plt.title('MTSL-RN和Thai-Hoang Pham基础模型的调和平均数对比')  # 图形标题
    pl.xticks(rotation=90)  # rotation the x-axis labels
    # plt.locator_params(,nbins=15)
    plt.show()
