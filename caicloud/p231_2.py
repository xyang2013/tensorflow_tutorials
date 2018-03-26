import tensorflow as tf

tf.reset_default_graph()

with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input1")

with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")

output = tf.add_n([input1, input2], name="add")

writer = tf.summary.FileWriter("/home/xyang/temp/log", tf.get_default_graph())
writer.close()
