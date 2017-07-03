import tensorflow as tf


def inception_unit(inputdata, weights, biases):
    # A3 inception 3a
    inception_in = inputdata

    # Conv 1x1+S1
    inception_1x1_S1 = tf.nn.conv2d(inception_in, weights['inception_1x1_S1'], strides=[1,1,1,1], padding='SAME')
    inception_1x1_S1 = tf.nn.bias_add(inception_1x1_S1, biases['inception_1x1_S1'])
    inception_1x1_S1 = tf.nn.relu(inception_1x1_S1)
    # Conv 3x3+S1
    inception_3x3_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_3x3_S1_reduce'], strides=[1,1,1,1], padding='SAME')
    inception_3x3_S1_reduce = tf.nn.bias_add(inception_3x3_S1_reduce, biases['inception_3x3_S1_reduce'])
    inception_3x3_S1_reduce = tf.nn.relu(inception_3x3_S1_reduce)
    inception_3x3_S1 = tf.nn.conv2d(inception_3x3_S1_reduce, weights['inception_3x3_S1'], strides=[1,1,1,1], padding='SAME')
    inception_3x3_S1 = tf.nn.bias_add(inception_3x3_S1, biases['inception_3x3_S1'])
    inception_3x3_S1 = tf.nn.relu(inception_3x3_S1)
    # Conv 5x5+S1
    inception_5x5_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_5x5_S1_reduce'], strides=[1,1,1,1], padding='SAME')
    inception_5x5_S1_reduce = tf.nn.bias_add(inception_5x5_S1_reduce, biases['inception_5x5_S1_reduce'])
    inception_5x5_S1_reduce = tf.nn.relu(inception_5x5_S1_reduce)
    inception_5x5_S1 = tf.nn.conv2d(inception_5x5_S1_reduce, weights['inception_5x5_S1'], strides=[1,1,1,1], padding='SAME')
    inception_5x5_S1 = tf.nn.bias_add(inception_5x5_S1, biases['inception_5x5_S1'])
    inception_5x5_S1 = tf.nn.relu(inception_5x5_S1)
    # MaxPool
    inception_MaxPool = tf.nn.max_pool(inception_in, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    inception_MaxPool = tf.nn.conv2d(inception_MaxPool, weights['inception_MaxPool'], strides=[1,1,1,1], padding='SAME')
    inception_MaxPool = tf.nn.bias_add(inception_MaxPool, biases['inception_MaxPool'])
    inception_MaxPool = tf.nn.relu(inception_MaxPool)
    # Concat
    #tf.concat(concat_dim, values, name='concat')
    #concat_dim 是 tensor 连接的方向（维度）， values 是要连接的 tensor 链表， name 是操作名。 cancat_dim 维度可以不一样，其他维度的尺寸必须一样。
    inception_out = tf.concat(concat_dim=3, values=[inception_1x1_S1, inception_3x3_S1, inception_5x5_S1, inception_MaxPool])
    return inception_out
​
def GoogleLeNet_topological_structure(x, weights, biases, conv_weights_3a, conv_biases_3a, conv_weights_3b, conv_biases_3b ,
                conv_weights_4a, conv_biases_4a, conv_weights_4b, conv_biases_4b, 
                conv_weights_4c, conv_biases_4c, conv_weights_4d, conv_biases_4d,
                conv_weights_4e, conv_biases_4e, conv_weights_5a, conv_biases_5a,
                conv_weights_5b, conv_biases_5b, dropout=0.8):
    # A0 输入数据
    x = tf.reshape(x,[-1,224,224,4])  # 调整输入数据维度格式

    # A1  Conv 7x7_S2
    x = tf.nn.conv2d(x, weights['conv1_7x7_S2'], strides=[1,2,2,1], padding='SAME')
    # 卷积层 卷积核 7*7 扫描步长 2*2 
    x = tf.nn.bias_add(x, biases['conv1_7x7_S2'])
    #print (x.get_shape().as_list())
    # 偏置向量
    x = tf.nn.relu(x)
    # 激活函数
    x = tf.nn.max_pool(x, ksize=pooling['pool1_3x3_S2'], strides=[1,2,2,1], padding='SAME')
    # 池化取最大值
    x = tf.nn.local_response_normalization(x, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta= 0.75)
    # 局部响应归一化

    # A2
    x = tf.nn.conv2d(x, weights['conv2_1x1_S1'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_1x1_S1'])
    x = tf.nn.conv2d(x, weights['conv2_3x3_S1'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_3x3_S1'])
    x = tf.nn.local_response_normalization(x, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta= 0.75)
    x = tf.nn.max_pool(x, ksize=pooling['pool2_3x3_S2'], strides=[1,2,2,1], padding='SAME')

    # inception 3
    inception_3a = inception_unit(inputdata=x, weights=conv_W_3a, biases=conv_B_3a)
    inception_3b = inception_unit(inception_3a, weights=conv_W_3b, biases=conv_B_3b)

    # 池化层
    x = inception_3b
    x = tf.nn.max_pool(x, ksize=pooling['pool3_3x3_S2'], strides=[1,2,2,1], padding='SAME' )

    # inception 4
    inception_4a = inception_unit(inputdata=x, weights=conv_W_4a, biases=conv_B_4a)
    # 引出第一条分支
    #softmax0 = inception_4a
    inception_4b = inception_unit(inception_4a, weights=conv_W_4b, biases=conv_B_4b)    
    inception_4c = inception_unit(inception_4b, weights=conv_W_4c, biases=conv_B_4c)
    inception_4d = inception_unit(inception_4c, weights=conv_W_4d, biases=conv_B_4d)
    # 引出第二条分支
    #softmax1 = inception_4d
    inception_4e = inception_unit(inception_4d, weights=conv_W_4e, biases=conv_B_4e)

    # 池化
    x = inception_4e
    x = tf.nn.max_pool(x, ksize=pooling['pool4_3x3_S2'], strides=[1,2,2,1], padding='SAME' )

    # inception 5
    inception_5a = inception_unit(x, weights=conv_W_5a, biases=conv_B_5a)
    inception_5b = inception_unit(inception_5a, weights=conv_W_5b, biases=conv_B_5b)
    softmax2 = inception_5b

    # 后连接
    softmax2 = tf.nn.avg_pool(softmax2, ksize=[1,7,7,1], strides=[1,1,1,1], padding='SAME')
    softmax2 = tf.nn.dropout(softmax2, keep_prob=0.4)
    softmax2 = tf.reshape(softmax2, [-1,weights['FC2'].get_shape().as_list()[0]])
    softmax2 = tf.nn.bias_add(tf.matmul(softmax2,weights['FC2']),biases['FC2'])
    #print(softmax2.get_shape().as_list())
    return softmax2  
​
weights = {
    'conv1_7x7_S2': tf.Variable(tf.random_normal([7,7,4,64])),
    'conv2_1x1_S1': tf.Variable(tf.random_normal([1,1,64,64])),
    'conv2_3x3_S1': tf.Variable(tf.random_normal([3,3,64,192])),
    'FC2': tf.Variable(tf.random_normal([7*7*1024, 3]))
}

biases = {
    'conv1_7x7_S2': tf.Variable(tf.random_normal([64])),
    'conv2_1x1_S1': tf.Variable(tf.random_normal([64])),
    'conv2_3x3_S1': tf.Variable(tf.random_normal([192])),
    'FC2': tf.Variable(tf.random_normal([3]))

}
pooling = {
    'pool1_3x3_S2': [1,3,3,1],
    'pool2_3x3_S2': [1,3,3,1],
    'pool3_3x3_S2': [1,3,3,1],
    'pool4_3x3_S2': [1,3,3,1]
}
​
​
​
conv_W_3a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,192,64])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,192,96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,96,128])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,192,16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,16,32])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,192,32]))

}
​
conv_B_3a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([64])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([128])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([32])),
    'inception_MaxPool': tf.Variable(tf.random_normal([32]))
}
​
conv_W_3b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,256,128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,256,128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,128,192])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,256,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,96])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,256,64]))

}
​
conv_B_3b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([192])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([96])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}
​
conv_W_4a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,480,192])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,480,96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,96,208])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,480,16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,16,48])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,480,64]))

}
​
conv_B_4a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([192])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([208])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([48])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}

​
conv_W_4b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,160])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,112])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,112,224])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,24,64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))

}
​
conv_B_4b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([160])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([112])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([224])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}
​
conv_W_4c = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,128,256])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,24,64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))

}
​
conv_B_4c = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([256])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}
​
conv_W_4d = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,112])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,144])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,144,288])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))

}
​
conv_B_4d = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([112])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([144])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([288])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}
​
conv_W_4e = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,528,256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,528,160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,160,320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,528,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,528,128]))

}
​
conv_B_4e = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([128]))
}
​
conv_W_5a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,832,256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,832,160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,160,320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,832,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,832,128]))

}
​
conv_B_5a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([128]))
}

conv_W_5b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,832,384])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,832,192])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,192,384])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,832,48])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,48,128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,832,128]))

}
​
conv_B_5b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([384])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([192])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([384])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([48])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([128]))
}