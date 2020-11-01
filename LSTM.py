# coding=gbk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#matplotlib inline
 
# 1. ����RNN�Ĳ���
HIDDEN_SIZE = 30                            # LSTM�����ؽڵ�ĸ�����
NUM_LAYERS = 2                              # LSTM�Ĳ�����
TIMESTEPS = 10                              # ѭ���������ѵ�����г��ȡ�
TRAINING_STEPS = 10000                      # ѵ��������
BATCH_SIZE = 32                             # batch��С��
TRAINING_EXAMPLES = 10000                   # ѵ�����ݸ�����
TESTING_EXAMPLES = 1000                     # �������ݸ�����
SAMPLE_GAP = 0.01                           # ���������
 
 
# 2. �����������ݺ���
def generate_data(seq):
    X = []
    y = []
    # ���еĵ�i��ͺ����TIMESTEPS-1�����һ����Ϊ���룻��i + TIMESTEPS����Ϊ�����
    # ����sin����ǰ���TIMESTEPS�������Ϣ��Ԥ���i + TIMESTEPS����ĺ���ֵ��
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  
 
 
# 3. ��������ṹ���Ż�����
def lstm_model(X, y, is_training):
    # ʹ�ö���LSTM�ṹ��
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)]) 
 
    # ʹ��TensorFlow�ӿڽ�����LSTM�ṹ���ӳ�RNN���粢������ǰ�򴫲������
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # outputs�Ƕ���LSTM��ÿһ����������������ά����[batch_size, time ,
    # HIDDEN_SIZE]���ڱ�������ֻ��ע���һ��ʱ�̵���������
    output = outputs[:, -1, :]
 
    # ��LSTM��������������һ��ȫ���Ӳ㲢������ʧ��ע������Ĭ�ϵ���ʧΪƽ��
    # ƽ������ʧ������
    predictions = tf.contrib.layers.fully_connected(
        output, 1, activation_fn=None)
    
    # ֻ��ѵ��ʱ������ʧ�������Ż����衣����ʱֱ�ӷ���Ԥ������
    if not is_training:
        return predictions, None, None
        
    # ������ʧ������
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
 
    # ����ģ���Ż������õ��Ż����衣
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)
    
    return predictions, loss, train_op
 
 
# 4. ����ѵ������
def train(sess, train_X, train_Y):
    # ��ѵ�����������ݼ��ķ�ʽ�ṩ������ͼ
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)   #maxi:���ָ���ô���
    X, y = ds.make_one_shot_iterator().get_next()
    
    # ����ģ�ͣ��õ�Ԥ��������ʧ��������ѵ��������
    with tf.variable_scope("model"):
        _, loss, train_op = lstm_model(X, y, True)
        
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 1000 == 0:
            print("train step: " + str(i) + ", loss: ", str(l))
            
 
# 5. ������Է���
def run_eval(sess, test_X, test_y):
    # ���������������ݼ��ķ�ʽ�ṩ������ͼ��
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    
    # ����ģ�͵õ������������ﲻ��Ҫ������ʵ��yֵ��
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)
    
    # ��Ԥ��������һ�����顣
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)
 
    # ����rmse��Ϊ����ָ�ꡣ
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Root Mean Square Error is: %f" % rmse)
    
    # ��Ԥ���sin�������߽��л�ͼ��
    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()
    
    
# 6. �������ݲ�ѵ������֤
# �����Һ�������ѵ���Ͳ������ݼ��ϡ�
# numpy.linspace�������Դ���һ���Ȳ����е����飬�����õĲ���������������
# ��һ��������ʾ��ʼֵ���ڶ���������ʾ��ֵֹ��������������ʾ���еĳ��ȡ�
# ����linespace(1, 10, 10)������������arrray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 
test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
 
#maxi:���Կ������Ĳ��������ѵ��ģ�ͣ�Ȼ���ѵ������ģ����ȥԤ��
with tf.Session() as sess:
    train(sess, train_X, train_y)
    run_eval(sess, test_X, test_y)