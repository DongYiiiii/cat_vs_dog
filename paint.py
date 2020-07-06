'''
@Author:Dongyi
@Date:2020.7.2
@Description:这是对实验的到的结果的绘图部分
这部分需要绘制包括accuracy，precision，roc
'''
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensor import InputData
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score

# 全局变量声明
IMAGE_SIZE = 100
LEARNING_RATE = 1e-4
TRAIN_STEP = 200
TRAIN_SIZE = 100
TEST_STEP = 100
TEST_SIZE = 50
data_dir = './batch_files'
pic_path = './data/test1'


# 首先读取h5文件中的参数
with h5py.File('record/pure_cnn.h5','r') as f:
    train_loss = f['train_loss'][:]
    train_accuracy = f['train_accuracy'][:]
    test_loss = f['test_loss'][:]
    test_accuracy = f['test_accuracy'][:]

    index = [10*i for i in range(100)]
    test_loss=test_loss[index]
    test_accuracy=test_accuracy[index]


# 绘制train test loss accuracy曲线
plt.plot(np.arange(100), train_loss, 'b', label='TrainLoss')
plt.plot(np.arange(100), train_accuracy, 'r', label='TrainAccuracy')
plt.plot(np.arange(100), test_loss, 'g', label='TestLoss')
plt.plot(np.arange(100), test_accuracy, 'm', label='TestAccuracy')
plt.xlabel('epochs')
plt.ylabel('loss/accuracy')
plt.title('Train & Test Loss/Accuracy')
plt.legend(loc='lower right')
plt.savefig('pure_cnn_train_test_loss_accuracy.png')
plt.show()
#
# # 接下来绘制纯cnn跑出来的roc曲线
# # 绘制roc曲线需要有预测值和label值
# # 这里将模型读取出来然后将对应预测层拿出，然后把预测值和label取出绘制roc曲线
# # 现在有一个问题就是sklearn绘制roc曲线时怎么样曲折一些（阈值不进行过滤）
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/my_model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model'))
    graph = tf.get_default_graph()
#
#     # 载入训练集和测试集
#     train_filenames = [os.path.join(data_dir, 'train_batch_%d' % i) for i in range(1, 101)]
#     test_filenames = [os.path.join(data_dir, 'test_batch')]
#     batch_train_data = InputData(train_filenames, True)
#     batch_test_data = InputData(test_filenames, True)
#
#     #输出模型预测的结果
#     softmax_predict = graph.get_tensor_by_name('Softmax:0')
#     x = graph.get_tensor_by_name('input_data:0')
#     y = graph.get_tensor_by_name('output_data:0')
#     keep_prob = graph.get_tensor_by_name('keep_prob:0')
#
#     #测试数据提取最后预测特征
#     final_predict_result = [[]]
#     all_test_label = []
#     for i in range(TEST_STEP):
#         test_data, test_label, _ = batch_test_data.next_batch(TEST_SIZE)
#         softmax_predict_result = sess.run(softmax_predict, feed_dict={
#             x: test_data,
#             y: test_label,
#             keep_prob: 1.0
#         })
#         final_predict_result += softmax_predict_result.tolist()
#         all_test_label += test_label.tolist()
#     del final_predict_result[0]
#     print(final_predict_result)
#
#     # 这里对final_predict_result中的数据进行处理，对每对值选择最大的下标记录
#     final_predict_result_probality = []
#     for i in range(len(final_predict_result)):
#         final_predict_result_probality.append(final_predict_result[i][1])
#
#     # 绘制roc曲线
#     false_postive_rate, true_positive_rate, thresholds = roc_curve(all_test_label, final_predict_result_probality)
#     roc_auc = auc(false_postive_rate, true_positive_rate)
#     plt.title('SVM classification ROC Curve')
#     plt.plot(false_postive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Postive Rate')
#     plt.savefig('pure_cnn_roc.png')
#     plt.show()



    '''
    以下代码是为了获取整个graph中图像的tensor名称
    '''
    tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
    for tensor_name in tensor_name_list:
        print(tensor_name, '\n')

    ## 查看网络每一层的参数：
    print('print the trainable parameters: ')
    for eval_ in tf.trainable_variables():
        print(eval_.name)
        w_val = sess.run(eval_.name)
        print(w_val.shape)
    op_list = sess.graph.get_operations()
    for op in op_list:
        print(op.name)
        print(op.values())




