from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import numpy as np
import tensorflow as tf

learn = tf.contrib.learn

def my_model(features, target):
    target = tf.one_hot(target, 3, 1, 0)
    logits, loss = learn.models.logistic_regression(features, target)
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adagrad',
        learning_rate = 0.1)
    return tf.arg_max(logits, 1), loss, train_op

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
x_train, x_test = map(np.float32, [x_train, x_test])

classifier = learn.Estimator(model_fn=my_model)

classifier.fit(x_train, y_train, steps=800)

y_predicted = classifier.predict(x_test)
y_predicted = [i for i in classifier.predict(x_test)]
print(y_predicted)

score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: %.2f%%' % (score*100))
