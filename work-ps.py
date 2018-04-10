# coding=utf-8
import tensorflow as tf
from time import sleep
# 现在假设我们有A、B两台机器，首先需要在各台机器上写一份代码，并跑起来，各机器上的代码内容大部分相同
# ，除了开始定义的时候，需要各自指定该台机器的task之外。以机器A为例子，A机器上的代码如下：
cluster = tf.train.ClusterSpec({
    "worker": [
        "localhost:1234",  # 格式 IP地址：端口号，第一台机器A的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:0
    ],
    "ps": [
        "192.168.1.120:2223"  # 第二台机器的IP地址 对应到代码块：/job:ps/task:0
    ]})

# 不同的机器，下面这一行代码各不相同，server可以根据job_name、task_index两个参数，查找到集群cluster中对应的机器

isps = False
if isps:
    server = tf.train.Server(cluster, job_name='ps', task_index=0)  # 找到‘worker’名字下的，task0，也就是机器A
    server.join()
else:
    server = tf.train.Server(cluster, job_name='worker', task_index=0)  # 找到‘worker’名字下的，task0，也就是机器A
    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:0', cluster=cluster)):
        w = tf.get_variable('w', (2, 2), tf.float32, initializer=tf.constant_initializer(2))
        b = tf.get_variable('b', (2, 2), tf.float32, initializer=tf.constant_initializer(5))
        addwb = w + b
        mutwb = w * b
        divwb = w / b

init_op = tf.global_variables_initializer()
with tf.Session(target=server.target) as sess:
    sess.run(init_op)
    while 1:
        print(sess.run([addwb, mutwb, divwb]))
        sleep(2)