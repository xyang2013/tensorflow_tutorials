import tensorflow as tf

a_cpu = tf.Variable(0, name="a_cpu")

with tf.device('/gpu:0'):
    a_gpu = tf.Variable(0, name="a_gpu")

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
