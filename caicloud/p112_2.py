import tensorflow as tf

saver = tf.train.import_meta_graph("/home/xyang/temp/model/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "/home/xyang/temp/model/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
