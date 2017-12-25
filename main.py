from sklearn.datasets import load_boston

from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import lstm

boston = load_boston()
x = boston.data
y = boston.target

print('波斯顿数据x:', x.shape)
print('波斯顿房价y:', y.shape)

ss_x = preprocessing.StandardScaler()
train_x = ss_x.fit_transform(x)
ss_y = preprocessing.StandardScaler()
train_y = ss_y.fit_transform(y.reshape(-1, 1))

BATCH_START = 0
TIME_STEPS = 10
BATCH_SIZE = 30
INPUT_SIZE = 13
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006

def get_batch_boston():
    global train_x, train_y, BATCH_START, TIME_STEPS
    x_part1 = train_x[BATCH_START : BATCH_START + TIME_STEPS * BATCH_SIZE]
    y_part1 = train_y[BATCH_START : BATCH_START + TIME_STEPS * BATCH_SIZE]
    #print('时间端=', BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE)

    seq = x_part1.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
    res = y_part1.reshape((BATCH_SIZE, TIME_STEPS, 1))

    BATCH_START += TIME_STEPS

    return [seq, res]

def get_batch():
    global BATCH_START, TIME_STEPS
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((
            BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
    print('xs.shape=', xs.shape)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPs
    print('增加维度前:', seq.shape)
    print(seq[:2])
    print('增加维度后:', seq[:, :, np.newaxis].shape)
    print(seq[:2])
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

if __name__ == '__main__':
    seq, res = get_batch_boston()
    model = lstm.LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    sess.run(tf.global_variables_initializer())
    for j in range(200):
        pred_res=None
        for i in range(20):
            seq, res = get_batch_boston()
            if i == 0:
                feed_dict = {model.xs: seq, model.ys: res}
            else:
                feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        model.cell_init_state: state}
            _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)
            pred_res=pred

            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
        print('(0) cost: '.format(j), round(cost, 4))
        BATCH_START=0
    print('结果:', pred_res.shape)
    print('实际:', train_y.flatten().shape)

    r_size = BATCH_SIZE * TIME_STEPS

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 3))
    axes = fig.add_subplot(1, 1, 1)
    line1, =axes.plot(range(100), pred.flatten()[-100:], 'b--', label='rnn计算结果')
    line3, =axes.plot(range(100), train_y.flatten()[-100:], 'r', label='实际')

    axes.grid()
    fig.tight_layout()
    plt.legend(handles=[line1, line3])
    plt.title('递归神经网络')
    plt.show()
