'''
@Author:Dongyi
@Date：2020.6.27
@Description:这是利用预训练模型对数据进行提取特征的过程
'''
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score
from sklearn import metrics
from xgboost import XGBClassifier

from tensor import MyTensor, InputData
from sklearn import svm

IMAGE_SIZE = 100
LEARNING_RATE = 1e-4
TRAIN_STEP = 200
TRAIN_SIZE = 100
TEST_STEP = 100
TEST_SIZE = 50

data_dir = './batch_files'
pic_path = './data/test1'

# 这里插入一个ELM的分类器
class SingeHiddenLayer(object):

    def __init__(self, X, y, num_hidden):
        self.data_x = np.atleast_2d(X)  # 判断输入训练集是否大于等于二维; 把x_train()取下来
        self.data_y = np.array(y).flatten()  # a.flatten()把a放在一维数组中，不写参数默认是“C”，也就是先行后列的方式，也有“F”先列后行的方式； 把 y_train取下来
        self.num_data = len(self.data_x)  # 训练数据个数
        self.num_feature = self.data_x.shape[
            1];  # shape[] 读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度 (120行，4列，所以shape[0]==120,shapep[1]==4)
        self.num_hidden = num_hidden;  # 隐藏层节点个数

        # 随机生产权重（从-1，到1，生成（num_feature行,num_hidden列））
        self.w = np.random.uniform(-1, 1, (self.num_feature, self.num_hidden))

        # 随机生成偏置，一个隐藏层节点对应一个偏置
        for i in range(self.num_hidden):
            b = np.random.uniform(-0.6, 0.6, (1, self.num_hidden))
            self.first_b = b

        # 生成偏置矩阵，以隐藏层节点个数4为行，样本数120为列
        for i in range(self.num_data - 1):
            b = np.row_stack((b, self.first_b))  # row_stack 以叠加行的方式填充数组
        self.b = b

    # 定义sigmoid函数
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def train(self, x_train, y_train, classes):
        mul = np.dot(self.data_x, self.w)  # 输入乘以权重
        add = mul + self.b  # 加偏置
        H = self.sigmoid(add)  # 激活函数

        H_ = np.linalg.pinv(H)  # 求广义逆矩阵
        # print(type(H_.shape))

        # 将只有一列的Label矩阵转换，例如，iris的label中共有三个值，则转换为3列，以行为单位，label值对应位置标记为1，其它位置标记为0
        self.train_y = np.zeros((self.num_data, classes))  # 初始化一个120行，3列的全0矩阵
        for i in range(0, self.num_data):
            self.train_y[i, y_train[i]] = 1  # 对应位置标记为1

        self.out_w = np.dot(H_, self.train_y)  # 求输出权重

    def predict(self, x_test):
        self.t_data = np.atleast_2d(x_test)  # 测试数据集
        self.num_tdata = len(self.t_data)  # 测试集的样本数
        self.pred_Y = np.zeros((x_test.shape[0]))  # 初始化

        b = self.first_b

        # 扩充偏置矩阵，以隐藏层节点个数4为行，样本数30为列
        for i in range(self.num_tdata - 1):
            b = np.row_stack((b, self.first_b))  # 以叠加行的方式填充数组

        # 预测
        self.pred_Y = np.dot(self.sigmoid(np.dot(self.t_data, self.w) + b), self.out_w)

        # 取输出节点中值最大的类别作为预测值
        self.predy = []
        for i in self.pred_Y:
            L = i.tolist()
            self.predy.append(L.index(max(L)))
        return self.pred_Y

    def score(self, y_test):
        print("准确率：")
        # 使用准确率方法验证
        print(metrics.accuracy_score(y_true=y_test, y_pred=self.predy))


#======================================正文=============================#
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/my_model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model'))
    graph = tf.get_default_graph()


    #这里要把所有的相关变量都创建出来
    flatten_feature = graph.get_tensor_by_name('flatten/Reshape:0')
    dense_layer1 = graph.get_tensor_by_name('dense/Relu:0') #512维向量
    dense_layer2 = graph.get_tensor_by_name('dense_1/Relu:0') #256维向量
    x = graph.get_tensor_by_name('input_data:0')
    y = graph.get_tensor_by_name('output_data:0')
    softmax_predict = graph.get_tensor_by_name('Softmax:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    #载入训练集和测试集
    train_filenames = [os.path.join(data_dir, 'train_batch_%d' % i) for i in range(1, 101)]
    test_filenames = [os.path.join(data_dir, 'test_batch')]
    batch_train_data = InputData(train_filenames, True)
    batch_test_data = InputData(test_filenames, True)

    # 输出训练数据的特征
    all_result = [[]]
    # all_dense_layer1_train = [[]]
    # all_dense_layer2_train = [[]]
    all_label = []
    for i in range(TRAIN_STEP):
        train_data, train_label, _ = batch_train_data.next_batch(TRAIN_SIZE)
        result = sess.run(flatten_feature, feed_dict={
            x: train_data,
            y: train_label,
            keep_prob: 0.7
        })
        # dense_layer1_train = sess.run(dense_layer1, feed_dict={
        #     x: train_data,
        #     y: train_label,
        #     keep_prob: 0.7
        # })
        # dense_layer2_train = sess.run(dense_layer2, feed_dict={
        #     x: train_data,
        #     y: train_label,
        #     keep_prob: 0.7
        # })
        all_result += result.tolist()
        # all_dense_layer1_train += dense_layer1_train.tolist()
        # all_dense_layer2_train += dense_layer2_train.tolist()
        all_label += train_label.tolist()
    del all_result[0]
    # del all_dense_layer1_train[0]
    # del all_dense_layer2_train[0]

    # 提取测试数据特征
    # 测试数据提取最后预测特征
    all_test_result = [[]]
    # dense_layer1_result_list = [[]]
    # dense_layer2_result_list = [[]]
    all_test_label = []
    final_predict_result = [[]]
    for i in range(TEST_STEP):
        test_data, test_label, _ = batch_test_data.next_batch(TEST_SIZE)
        test_result = sess.run(flatten_feature, feed_dict={
            x: test_data,
            y: test_label,
            keep_prob: 1.0
        })
        # dense_layer1_result = sess.run(dense_layer1, feed_dict={
        #     x: test_data,
        #     y: test_label,
        #     keep_prob: 1.0
        # })
        # dense_layer2_result = sess.run(dense_layer2, feed_dict={
        #     x: test_data,
        #     y: test_label,
        #     keep_prob: 1.0
        # })
        softmax_predict_result = sess.run(softmax_predict, feed_dict={
            x: test_data,
            y: test_label,
            keep_prob: 1.0
        }) #这是对最后预测结果的提取
        all_test_result += test_result.tolist() # 这是对cnn-svm结果的提取，提取的向量是flatten之后的4608维度的向量
        # dense_layer1_result_list += dense_layer1_result.tolist()
        # dense_layer2_result_list += dense_layer2_result.tolist()
        all_test_label += test_label.tolist()
        final_predict_result += softmax_predict_result.tolist()
    del all_test_result[0]
    del final_predict_result[0]
    # del dense_layer1_result_list[0]
    # del dense_layer2_result_list[0]

    # 这里对final_predict_result中的数据进行处理，对每对值选择后者
    final_predict_result_probality = []
    for i in range(len(final_predict_result)):
        final_predict_result_probality.append(final_predict_result[i][1])
    # 这里记录最后的大值的下标
    final_predict_result_index = []
    for i in range(len(final_predict_result)):
        if final_predict_result[i][0] < final_predict_result[i][1]:
            final_predict_result_index.append(1)
        else:
            final_predict_result_index.append(0)

    #sklearn svm二分类
    input_x = all_result
    label_x = all_label
    input_test_x = all_test_result
    clf = svm.SVC(probability=True)
    clf.fit(input_x,label_x)
    # predictions = clf.predict_proba(input_test_x)[:,1]
    predictions = clf.predict(input_test_x)

    #sklearn XGBoost二分类
    input_x2 = all_result
    label_x2 = all_label
    input_test_x2 = all_test_result
    model = XGBClassifier()
    model.fit(np.array(input_x2), np.array(label_x2))
    # xgb_predictions = model.predict_proba(np.array(input_test_x2))[:,1]
    xgb_predictions = model.predict(np.array(input_test_x2))

    # ELM二分类
    input_x3 = all_result
    label_x3 = all_label
    input_test_x3 = all_test_result
    ELM = SingeHiddenLayer(input_x3, label_x3, 512)
    ELM.train(input_x3, label_x3, 2)
    ELM_predictions = ELM.predict(np.array(input_test_x3))
    # 这里对ELM最后的预测值进行处理
    ELM_final_predict_result_probality = []
    ELM_final_predict_result_index = []
    for i in range(len(ELM_predictions)):
        ELM_final_predict_result_probality.append(ELM_predictions[i][1])
    # 记录最大值的下标
    for i in range(len(ELM_predictions)):
        if ELM_predictions[i][0] < ELM_predictions[i][1]:
            ELM_final_predict_result_index.append(1)
        else:
            ELM_final_predict_result_index.append(0)

    # ==================================以下是绘制选取不同维度的特征向量作为分类向量的情况===========================================#
    '''
    1.原始选择4608维度向量作为特征向量
    2.下一层选取512维度作为特征向量
    3.下一层选择256维度作为特征向量
    三者svm的性能比较
    '''
    # # 原始选择4608维度向量
    # input_x = all_result
    # label_x = all_label
    # input_test_x = all_test_result
    # clf = svm.SVC(probability=True)
    # clf.fit(input_x, label_x)
    # predictions = clf.predict_proba(input_test_x)[:, 1]
    # # 选择512维度向量
    # input_x2 = all_dense_layer1_train
    # label_x2 = all_label
    # input_test_x2 = dense_layer1_result_list
    # clf2 = svm.SVC(probability=True)
    # clf2.fit(input_x2, label_x2)
    # predictions2 = clf2.predict_proba(input_test_x2)[:,1]
    # # 选择256维度向量
    # input_x3 = all_dense_layer2_train
    # label_x3 = all_label
    # input_test_x3 = dense_layer2_result_list
    # clf3 = svm.SVC(probability=True)
    # clf3.fit(input_x3, label_x3)
    # predictions3 = clf3.predict_proba(input_test_x3)[:,1]
    #
    # #绘制cnn-svm 的roc评价曲线
    # #4608维度
    # false_postive_rate, true_positive_rate, thresholds = roc_curve(all_test_label, predictions)
    # roc_auc = auc(false_postive_rate, true_positive_rate)
    # plt.plot(false_postive_rate, true_positive_rate, 'm', label='cnn-svm with 4608 AUC = %0.2f'% roc_auc)
    # #512维度
    # false_postive_rate2, true_positive_rate2, thresholds2 = roc_curve(all_test_label, predictions2)
    # roc_auc = auc(false_postive_rate2, true_positive_rate2)
    # plt.plot(false_postive_rate2, true_positive_rate2, 'g', label='cnn-svm with 512 AUC = %0.2f'% roc_auc)
    # #256维度
    # false_postive_rate3, true_positive_rate3, thresholds3 = roc_curve(all_test_label, predictions3)
    # roc_auc = auc(false_postive_rate3, true_positive_rate3)
    # plt.plot(false_postive_rate2, true_positive_rate2, '', label='cnn-svm with 256 AUC = %0.2f'% roc_auc)
    #
    # plt.title('cnn-svm with differente feature dimension')
    # plt.legend(loc='lower right')
    # plt.savefig('cnn-svm with differente feature dimension.png')
    # plt.show()


#==========================================绘图&评价指标===============================================#
    # 绘制纯cnn的预测情况roc
    # false_postive_rate1, true_positive_rate1, thresholds1 = roc_curve(all_test_label, final_predict_result_probality)
    # roc_auc = auc(false_postive_rate1, true_positive_rate1)
    # plt.plot(false_postive_rate1, true_positive_rate1, 'b', label='pure cnn AUC = %0.2f' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # 打印纯cnn的各个评价指标
    print("PureCNN Accu: %0.2f" % accuracy_score(all_test_label, final_predict_result_index))
    print("Pure-SVM Recall: %0.2f" % recall_score(all_test_label, final_predict_result_index, average='micro'))
    print("Pure-SVM Precision: %0.2f" % precision_score(all_test_label, final_predict_result_index, average='micro'))
    print("Pure-SVM F1: %0.2f" % f1_score(all_test_label, final_predict_result_index, average='micro'))


    #绘制cnn-svm 的roc评价曲线
    # false_postive_rate2, true_positive_rate2, thresholds2 = roc_curve(all_test_label, predictions)
    # roc_auc = auc(false_postive_rate2, true_positive_rate2)
    # plt.title('SVM classification ROC Curve')
    # plt.plot(false_postive_rate2, true_positive_rate2, 'm', label='cnn-svm AUC = %0.2f'% roc_auc)
    #打印cnn-svm的各个评价指标
    print("CNN-SVM Accu: %0.2f"% accuracy_score(all_test_label, predictions))
    print("CNN-SVM Recall: %0.2f"% recall_score(all_test_label, predictions, average='micro'))
    print("CNN-SVM Precision: %0.2f"% precision_score(all_test_label, predictions, average='micro'))
    print("CNN-SVM F1: %0.2f"% f1_score(all_test_label, predictions, average='micro'))


    # # 绘制xgboost roc评价曲线
    # false_postive_rate3, true_positive_rate3, thresholds3 = roc_curve(all_test_label, xgb_predictions)
    # roc_auc = auc(false_postive_rate3, true_positive_rate3)
    # plt.plot(false_postive_rate3, true_positive_rate3, 'g', label='cnn-xgboost AUC = %0.2f' % roc_auc)
    #
    # 打印cnn-xgboost的各个评价指标
    print("CNN-XGBoost Accu: %0.2f" % accuracy_score(all_test_label, xgb_predictions))
    print("CNN-XGBoost Recall: %0.2f" % recall_score(all_test_label, xgb_predictions, average='micro'))
    print("CNN-XGBoost Precision: %0.2f" % precision_score(all_test_label, xgb_predictions, average='micro'))
    print("CNN-XGBoost F1: %0.2f" % f1_score(all_test_label, xgb_predictions, average='micro'))

    # # 绘制cnn-elm 的 roc评价曲线
    # false_postive_rate4, true_positive_rate4, thresholds4 = roc_curve(all_test_label, ELM_final_predict_result_probality)
    # roc_auc = auc(false_postive_rate4, true_positive_rate4)
    # plt.plot(false_postive_rate4, true_positive_rate4, 'y', label='cnn-elm AUC = %0.2f' % roc_auc)
    # 打印cnn-elm的各个评价指标
    print("CNN-elm Accu: %0.2f" % accuracy_score(all_test_label, ELM_final_predict_result_index))
    print("CNN-elm Recall: %0.2f" % recall_score(all_test_label, ELM_final_predict_result_index, average='micro'))
    print("CNN-elm Precision: %0.2f" % precision_score(all_test_label, ELM_final_predict_result_index, average='micro'))
    print("CNN-elm F1: %0.2f" % f1_score(all_test_label, ELM_final_predict_result_index, average='micro'))

    # plt.legend(loc='lower right')
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Postive Rate')
    # plt.savefig('total_roc.png')
    # plt.show()













#===============================下面是在提取特征时的查看tensor形状的函数==============================#
    # tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
    # for tensor_name in tensor_name_list:
    #     print(tensor_name,'\n')

    # ## 查看网络每一层的参数：
    # print('print the trainable parameters: ')
    # for eval_ in tf.trainable_variables():
    #     print(eval_.name)
    #     w_val = sess.run(eval_.name)
    #     print(w_val.shape)

