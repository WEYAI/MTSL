import os
import matplotlib.pyplot as plt
import numpy as np

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入统计数据
waters = ('分词 \n（Conll 2003）', '词性标注 \n (Conll 2003)')
buy_number_male = [6, 7]
buy_number_female = [9, 4]
# ValueError: shape mismatch: objects cannot be broadcast to a single shape
# 可能是条形图的x和y的数组长度不同造成的，需要修改数据，保持一致
bar_width = 0.3  # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标

# 使用两次 bar 函数画出两组条形图
plt.bar(index_male, height=buy_number_male, width=bar_width, color='b', label='分词')
plt.bar(index_female, height=buy_number_female, width=bar_width, color='g', label='词性标注')
plt.bar(index_male, height=buy_number_male, width=bar_width, color='w', label='命名实体识别')

plt.legend()  # 显示图例
plt.xticks(index_male + bar_width / 2, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('调和平均数')  # 纵坐标轴标题
plt.title('MTSL-RN和Thai-Hoang Pham基础模型的调和平均数对比')   # 图形标题
plt.savefig(os.path.join('./', 'F1'+'.png'))

plt.show()
