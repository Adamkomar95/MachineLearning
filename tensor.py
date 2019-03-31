import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

# x = 0.0
# y = 0.0
# tx=[]
# ty=[]
# ITER_NUM = 1000000
#
def przeksz(x,y,r):
    if r == 1:
        x_i = 0.0
        y_i = (0.16 * y)
    if r >= 2 and r <= 86:
        x_i = (0.85 * x + 0.04 * y)
        y_i = (-0.04 * x + 0.85 * y + 1.6)
    if r >= 87 and r <= 93:
        x_i = (0.20 * x - 0.26 * y)
        y_i = (0.23 * x + 0.22 * y + 1.6)
    if r >= 94 and r <= 100:
        x_i = (-0.15 * x + 0.28 * y)
        y_i = (0.26 * x + 0.24 * y + 0.44)
    return x_i,y_i
#
# for i in range(ITER_NUM):
#     r = random.randint(1,100)
#     x,y = przeksz(x,y,r)
#     #y = przeksz_y(y,r)
#     tx.append(x)
#     ty.append(y)
#
# plt.plot(tx,ty,'go',markersize=0.01)
# plt.show()

##KONIEC WERSJI ZWYKŁEJ

##TENSORFLOW

# def przeksz(x,y,r):
#     if r == 1:
#         x_i = 0.0
#         y_i = (0.16 * y)
#     if r >= 2 and r <= 86:
#         x_i = (0.85 * x + 0.04 * y)
#         y_i = (-0.04 * x + 0.85 * y + 1.6)
#     if r >= 87 and r <= 93:
#         x_i = (0.20 * x - 0.26 * y)
#         y_i = (0.23 * x + 0.22 * y + 1.6)
#     if r >= 94 and r <= 100:
#         x_i = (-0.15 * x + 0.28 * y)
#         y_i = (0.26 * x + 0.24 * y + 0.44)
#     return x_i,y_i
#
#
# in_x = tf.placeholder(tf.float64)
# in_y = tf.placeholder(tf.float64)
# in_r = tf.placeholder(tf.int32)
#
# zero = np.zeros(1)
# tx= tf.Variable(np.zeros(1), name='tx',dtype=tf.float64)
# ty = tf.Variable(np.zeros(1), name='ty',dtype=tf.float64)
# ITER_NUM = 1000
#
# tf_przeksz = tf.py_func(przeksz,[in_x,in_y,in_r],(tf.float64,tf.float64))
#
# with tf.Session() as session:
#     r = random.randint(1,100)
#     tensor_tx = tf.convert_to_tensor(zero, dtype="float64")
#     tensor_ty = tf.convert_to_tensor(zero, dtype="float64")
#     session.run(tf.group(tensor_tx,tensor_ty))
#     x,y = session.run(tf_przeksz,feed_dict={in_x:0.0,in_y:0.0,in_r:r})
#     for i in range(ITER_NUM):
#         r = random.randint(1, 100)
#         (x,y)= session.run(tf_przeksz, feed_dict={in_x: x, in_y: y, in_r: r})
#         iks = tf.cast(x,dtype=tf.float64)
#         igrek = tf.cast(y,dtype=tf.float64)
#         session.run(tf.group(iks,igrek))
#         ten_tx = tf.concat([tensor_tx,[iks]],axis=0)
#         ten_ty = tf.concat([tensor_ty,[igrek]],axis=0)
#         tensor_tx =session.run(ten_tx)
#         tensor_ty =session.run(ten_ty)
#         print(i)
# plt.plot(tensor_tx,tensor_ty,'go',markersize=1)
# plt.show()

##TEST

def przeksz(x,y,r):
    if r == 1:
        x_i = 0.0
        y_i = (0.16 * y)
    if r >= 2 and r <= 86:
        x_i = (0.85 * x + 0.04 * y)
        y_i = (-0.04 * x + 0.85 * y + 1.6)
    if r >= 87 and r <= 93:
        x_i = (0.20 * x - 0.26 * y)
        y_i = (0.23 * x + 0.22 * y + 1.6)
    if r >= 94 and r <= 100:
        x_i = (-0.15 * x + 0.28 * y)
        y_i = (0.26 * x + 0.24 * y + 0.44)
    return x_i,y_i


in_x = tf.placeholder(tf.float64)
in_y = tf.placeholder(tf.float64)
in_r = tf.placeholder(tf.int32)

tx= [0.0]
ty =[0.0]
ITER_NUM = 1000000

tf_przeksz = tf.py_func(przeksz,[in_x,in_y,in_r],(tf.float64,tf.float64))

with tf.Session() as session:
    r = random.randint(1,100)
    x,y = session.run(tf_przeksz,feed_dict={in_x:0.0,in_y:0.0,in_r:r})
    for i in range(ITER_NUM):
        r = random.randint(1, 100)
        (x,y)= session.run(tf_przeksz, feed_dict={in_x: x, in_y: y, in_r: r})
        tx.append(x)
        ty.append(y)
        #print(i)

plt.plot(tx,ty,'go',markersize=0.1)
plt.show()



