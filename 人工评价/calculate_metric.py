import xlrd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from itertools import product
import numpy as np
import csv
from os.path import dirname, join

def plot_ROC_curve(all_labels, all_scores, num_classes, title):
    """
    输入:all_labels:数据的真实标签
        all_scores:输入数据的预测结果
        title:画出 ROC 图像的标题
    输出:figure:ROC曲线图像
    作用:绘制 ROC 曲线并计算 AUC 值
    """
    # 需注意绘制 ROC 曲线时,传入的 all_labels 必须转换为独热编码,all_socres 要转换为1维,元素代表取得评估概率（大于阈值为正例,否则为负例）
    figure = plt.figure()

    if all_labels.ndim != 1: # 多分类的情况
        raise ValueError('Do not support ndim != 1')
    else:
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)

        opt_idx = np.argmax(tpr-fpr) # 取距离正对角线最大的点
        opt_threshold = thresholds[opt_idx]
        opt_point = (fpr[opt_idx], tpr[opt_idx])

        cross_idx = np.argmin(np.abs(tpr+fpr-1)) # 取距离反对角线最近的点
        cross_threshold = thresholds[cross_idx]

        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标,真正率为纵坐标做曲线
        plt.plot(fpr[opt_idx], tpr[opt_idx], '*', color='deepskyblue', lw=lw, label="opt point[threshold={:.2f}, ({:.2f},{:.2f})]".format(opt_threshold, fpr[opt_idx], tpr[opt_idx]))
        plt.plot(fpr[cross_idx], tpr[cross_idx], '*', color='g', lw=lw, label="diag point[threshold={:.2f}, ({:.2f},{:.2f})]".format(cross_threshold, fpr[cross_idx], tpr[cross_idx]))
        # plt.text(fpr[opt_idx], tpr[opt_idx], '(%.2f,%.2f)' % (fpr[opt_idx], tpr[opt_idx]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.plot([1, 0], [0, 1], color='silver', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
    
    return roc_auc, figure, opt_threshold, opt_point

def plot_confusion_matrix(all_labels, all_preds, class_names, title):
    """
    输入:cm (array, shape = [n, n]):混淆矩阵
        class_names (array, shape = [n]):分类任务中类别的名字
        title (string):生成图片的标题
    输出:figure:混淆矩阵可视化图片对象
    """
    cm = confusion_matrix(all_labels, all_preds)
    figure = plt.figure(figsize=[6.4,5.0])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.array(range(len(class_names)))    
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names, rotation=-45)

    # 将混淆矩阵的数值标准化(保留小数点后两位),即每一个元素除以矩阵中每一行(真实标签)元素之和
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    thresh = cm.max() / 2.
    # 此处遍历为按照生成的笛卡尔集顺序遍历
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment='center',
                color='white' if cm[i,j] > thresh else 'black',
                fontsize=16)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure, cm


exel_path = 'C:\\Users\\yujun\\Desktop\\肛门括约肌人工识别-后\\肛门括约肌人工质控结果登记8.xlsx'
doctor_name = 'doctor8'
wb = xlrd.open_workbook(exel_path)

# test1
test1_sheet = wb.sheet_by_name('测试集1')

test1_img_ids = np.array([int(item) for item in test1_sheet.col_values(0)[1:]])
test1_man_predicts = np.array([int(item) for item in test1_sheet.col_values(1)[1:]])
test1_man_confidences = np.array([int(item)*0.2 for item in test1_sheet.col_values(2)[1:]])

test1_target = []
with open('C:\\Users\\yujun\\Desktop\\肛门括约肌人工识别-后\\前瞻性测试集_打乱.csv', newline='\n') as shuffle_csv:
    csv_reader = csv.reader(shuffle_csv)
    next(csv_reader)
    for row in csv_reader:
        test1_target.append(int(row[1]))
test1_target = np.array(test1_target)

cm_name = doctor_name+"_test1_Confusion_Matrix"
cm_figure, cm = plot_confusion_matrix(test1_target, test1_man_predicts, ['standard', 'nonstandard'], cm_name)
cm_figure.savefig(join(dirname(exel_path), cm_name+'.png'), bbox_inches='tight')

roc_name = doctor_name+"_test1_ROC_Curve"
roc_auc, roc_figure, opt_threshold, opt_point = plot_ROC_curve(test1_target, test1_man_confidences, ['standard', 'nonstandard'], roc_name)
roc_figure.savefig(join(dirname(exel_path), roc_name+'.png'), bbox_inches='tight')

acc = np.sum(test1_target == test1_man_predicts) / len(test1_target)
tn, fp, fn, tp = cm.ravel()
recall = tp / (tp + fn)
precision = tp / (tp + fp)
specificity = tn / (tn + fp)

print('Test1 => acc = {:.4f} , auc = {:.4f} , precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}, opt_threshold = {}, opt_point = {}'.format(
                    acc, roc_auc, precision, recall, specificity, opt_threshold, opt_point))


# test2
test2_sheet = wb.sheet_by_name('测试集2')

test2_img_ids = np.array([int(item) for item in test2_sheet.col_values(0)[1:]])
test2_man_predicts = np.array([int(item) for item in test2_sheet.col_values(1)[1:]])
test2_man_confidences = np.array([int(item)*0.2 for item in test2_sheet.col_values(2)[1:]])

test2_target = []
with open('C:\\Users\\yujun\\Desktop\\肛门括约肌人工识别-后\\湖南测试集_打乱.csv', newline='\n') as shuffle_csv:
    csv_reader = csv.reader(shuffle_csv)
    next(csv_reader)
    for row in csv_reader:
        test2_target.append(int(row[1]))
test2_target = np.array(test2_target)

cm_name = doctor_name+"_test2_Confusion_Matrix"
cm_figure, cm = plot_confusion_matrix(test2_target, test2_man_predicts, ['standard', 'nonstandard'], cm_name)
cm_figure.savefig(join(dirname(exel_path), cm_name+'.png'), bbox_inches='tight')

roc_name = doctor_name+"_test2_ROC_Curve"
roc_auc, roc_figure, opt_threshold, opt_point = plot_ROC_curve(test2_target, test2_man_confidences, ['standard', 'nonstandard'], roc_name)
roc_figure.savefig(join(dirname(exel_path), roc_name+'.png'), bbox_inches='tight')

acc = np.sum(test2_target == test2_man_predicts) / len(test2_target)
# print(cm)
tn, fp, fn, tp = cm.ravel()
recall = tp / (tp + fn)
precision = tp / (tp + fp)
specificity = tn / (tn + fp)

print('Test2 => acc = {:.4f} , auc = {:.4f} , precision = {:.4f} , recall = {:.4f} , specificity = {:.4f}, opt_threshold = {}, opt_point = {}'.format(
                    acc, roc_auc, precision, recall, specificity, opt_threshold, opt_point))