
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pylab import mpl

# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
font = {'family': 'sans-serif',
        'color': 'k',
        'weight': 'normal',
        'size': 20, }


def visual(y_true, y_pred, label, set_name='NotoNotes5.0'):
    # get a confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=label)
    # 归一化
    # con_mat1 = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    # con_mat1 = np.around(con_mat1, decimals=2)

    data = {}
    for i in range(len(label)):
        # data[label[i]] = con_mat1[i]
        data[label[i]] = conf_mat[i]
    pd_data = pd.DataFrame(data, index=label, columns=label)
    print(pd_data)

    f, ax = plt.subplots(figsize=(10, 10))
    ax.xaxis.tick_top()
    # color = 'Blues'
    color = 'binary_r'
    # color = 'RdPu'
    ax = sns.heatmap(pd_data, ax=ax, annot=False, vmin=0, vmax=1000, cmap=color, cbar=True)  # 画heatmap，具体参数可以查文档
    # ax = sns.heatmap(pd_data, ax=ax, vmin=-0.1, vmax=1, cmap='Greys', cbar=False) #画heatmap，具体参数可以查文档
    # ax = sns.heatmap(pd_data, ax=ax, fmt='.1f', cmap='Greys_r') #画heatmap，具体参数可以查文档

    plt.xticks(label=label, fontsize=10)  # x轴刻度的字体大小（文本包含在pd_data中了）
    plt.yticks(label=label, fontsize=10, rotation=360)  # y轴刻度的字体大小（文本包含在pd_data中了）

    # 设置colorbar的刻度字体大小
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=10)
    # 设置colorbar的label文本和字体大小
    # cbar = ax.collections[0].colorbar
    # cbar.set_label(r'$NMI$',fontdict=font)
    # 保存
    plt.savefig(os.path.join('./', 'Confusion_Matrix_1_pos' + set_name + 'eng_' + color + '.png'))
    plt.show()
    plt.close()


if __name__ == '__main__':
    # y_true=[1,0,0,2,1,0,3,3,3]
    # y_pred=[1,1,0,2,1,0,1,3,3]

    y_true = []
    y_pred = []
    with open('test_pos198.txt', 'r+', encoding='utf-8') as file:
        line_lists = file.readlines()
        for line in line_lists:
            if line != '\n':
                line_tags = line.split()
                true_tag = line_tags[1]
                predict_tag = line_tags[2]
                y_true.append(true_tag)
                y_pred.append(predict_tag)
        tags_list = y_true + y_pred
        tags_type = set(tags_list)


    label = list(tags_type)
    visual(y_true, y_pred, label)
