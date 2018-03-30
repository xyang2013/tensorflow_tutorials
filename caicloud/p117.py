import tensorflow as tf

tf.reset_default_graph()

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result1 = v1 + v2

saver = tf.train.Saver()
saver.export_meta_graph("/home/xyang/temp/model/p117/model.ckpt.meta.json", as_text=True)
