import tensorflow as tf
import numpy as np
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

row_count =3000
iteration= 500;
HIDDEN_NODES = 50
learning_rate = 0.01

circleInfo1= (2,2,2)
circleInfo2 = (5,5,3)
circleInfo3 = (6,1,1)


def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)


x_data= np.random.rand(2,row_count)*10
y_data= np.zeros((row_count ,1))

print(" circleInfo1[0:2]", circleInfo1[0:2],'r',circleInfo1[2]  )

def check( x, y ):
  #  centerPos= circleInfo1[0:2]
    centerPos2= circleInfo2[0:2]
  #  centerPos3 = circleInfo3[0:2]

  #  dist1= distance.euclidean(centerPos, (x,y))

  #  dist3 = distance.euclidean(centerPos3, (x, y))

    centerPos2= circleInfo2[0:2]
    dist2 = distance.euclidean(centerPos2, (x, y))
    if( x >5 and x<7 ):
        return 1


    """
    if(dist1 <=circleInfo1[2] ):
        return 1

    if(dist2 <= circleInfo2[2]):
       return 1

    if(dist3 <=circleInfo3[2] ):
        return 1
    """

    return 0




for i in range(row_count):
    y_data[i][0]=  check(x_data[0][i] ,x_data[1][i])



print("x_data ","\n", x_data)
print("y_data ","\n", y_data)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
x_len= len(x_data)
print("xlen",x_len)


def w1hypo():
    W1 = tf.Variable(tf.random_uniform([x_len,1]))
    L1=tf.matmul(tf.transpose(X), W1)
    return tf.sigmoid(L1)

def w3hypo():
    W1 = tf.get_variable("W1", shape=[x_len, HIDDEN_NODES], initializer=xavier_init(x_len, HIDDEN_NODES))
    W2 = tf.get_variable("W2", shape=[HIDDEN_NODES, HIDDEN_NODES], initializer=xavier_init(HIDDEN_NODES, HIDDEN_NODES))
    W3 = tf.get_variable("W3", shape=[HIDDEN_NODES, 1], initializer=xavier_init(HIDDEN_NODES, 1))

    b1 = tf.Variable(tf.random_uniform([HIDDEN_NODES]))
    b2 = tf.Variable(tf.random_uniform([HIDDEN_NODES]))
    b3 = tf.Variable(tf.random_uniform([1]))

    L1 = tf.nn.relu(tf.matmul(tf.transpose(X), W1)+b1)
    L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)
    L3 = tf.matmul(L2, W3)+b3
    return tf.sigmoid(L3)

hypothesis =w3hypo()

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(hypothesis,Y)
loss = tf.reduce_mean(cross_entropy)

train= tf.train.AdamOptimizer(learning_rate).minimize(loss)
init =tf.global_variables_initializer()




with tf.Session() as sess:
    sess.run(init)
    feed_dict = {X: x_data, Y: y_data}

    correct_prediction =tf.floor(hypothesis + 0.5)
    isValid= tf.equal(correct_prediction, Y)
    accuracy = tf.reduce_mean(tf.cast(isValid, "float"))

    for step in range(iteration):
        _,out_loss,out_collect_prediction,out_accuracy=sess.run([train,loss,correct_prediction,accuracy], feed_dict= feed_dict )
        if(step%(iteration/5) ==0):
            print(step, out_loss,out_accuracy)


    boolean_prediction= tf.equal(correct_prediction,1)

    out_boolean_prediction=  sess.run(boolean_prediction, feed_dict=feed_dict)

    data = {"x": [], "y": [], "result": []}
    for i in range(len(out_boolean_prediction)):
        data["x"].append(x_data[0][i])
        data["y"].append(x_data[1][i])
        data["result"].append(out_boolean_prediction[i][0])


    df = pd.DataFrame(data)

    g= sns.lmplot("x", "y", data=df,size=7, hue="result" ,legend=True,  fit_reg=False)


    circleData= {"x":[
        circleInfo1[0],
        circleInfo2[0]
    ], "y":[
        circleInfo1[1],
        circleInfo2[1]
    ],"r":[
        circleInfo1[2],
        circleInfo2[2]
    ]
    }
    df_2 = pd.DataFrame(circleData)

   # plt.scatter(circleData["x"],circleData["y"],c=circleData["r"])
 #   circle1= plt.Circle((circleInfo1[0], circleInfo1[1]), circleInfo1[2], color='b', alpha=0.1)
 #   circle2= plt.Circle((circleInfo2[0], circleInfo2[1]), circleInfo2[2], color='b', alpha=0.1)
 #   ax = plt.gca()
 #   ax.add_artist(circle1)
 #   ax.add_artist(circle2)

    plt.show()


    test_x = 1
    test_y = 1
    print("Test ",test_x ," ",test_y, tf.floor(hypothesis + 0.5).eval(feed_dict={X: [[test_x],[test_y]] }) )