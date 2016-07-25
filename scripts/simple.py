import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops

logits = tf.placeholder( tf.float32,shape=[3,2], name='logits' )

targets = tf.placeholder( tf.float32,shape=[3,2], name='targets' )

loss = tf.nn.softmax_cross_entropy_with_logits( logits, targets )

predictions = tf.nn.softmax(logits)

iters = tf.placeholder( tf.float32, shape=[1], name='iters' )

cost = tf.reduce_sum(loss)
error = 1.0 - tf.reduce_sum( tf.mul( tf.floor( predictions + 0.5 ), targets ) ) / iters[0]

init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

L = [ [0.0,0.0],[0.4,0.8],[0.1,0.9] ]

T = [ [0.0,0.0],[0.0,1.0],[1.0,0.0] ]

I = [ 2 ]

feed_dict = dict()

feed_dict[logits]  = L
feed_dict[targets] = T
feed_dict[iters]   = I

lss,cst,err,pred = sess.run( [loss,cost,error,predictions], feed_dict )

print("logits")
print(L)
print("targets")
print(T)
print("preds")
print(pred)
print("loss")
print(lss)
print("-"*10)
print("cost:  %.6f"%cst)
print("error: %.6f"%err)

