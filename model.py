import tensorflow as tf 
import numpy as np 
import time
from data import Data 

class Network():

	def __init__(self, batch_size, feature_len, class_num, group):

		self.batch_size = batch_size
		self.feature_len = feature_len
		self.class_num = class_num
		self.group = group


	def create_network(self, input_x):

		#Initializing Parameters to be learnt
		w = tf.get_variable("weights", shape = [self.feature_len, self.class_num], initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-4)) #(512, 1595)
		b  = tf.get_variable("biases", shape = [self.class_num], initializer = tf.constant_initializer(0.0001)) #(1595,)
		q_param = tf.get_variable("q0", shape = [self.feature_len], initializer = tf.constant_initializer(0.0001)) #(512,)

		

		#attention module 
		resize_input = tf.reshape(input_x, [self.batch_size * self.group, self.feature_len]) # (640, 512)
		expand_param = tf.expand_dims(q_param, 1) # (512, 1)

		ek = tf.matmul(resize_input, expand_param) # (640,512) * (512,1) = (640,1).  
		ek = tf.reshape(ek, [self.batch_size, self.group]) #(128, 5). These are corresponding significances of each input feature based on quality of image
		ak = tf.nn.softmax(ek) #(128, 5). Changing them into positive values where their sum totals to 1

		features = tf.split(axis = 0, num_or_size_splits = self.batch_size, value = input_x) # A list of len = 128.  Each element is of shape (1, 5, 512)
		temps = tf.split(axis = 0, num_or_size_splits = self.batch_size, value = ak) # A list of len = 128. Each element is of shape (1, 5)
		fusion = [tf.matmul(temps[i], features[i][0]) for i in range(self.batch_size)] # (1,5) * (5, 512) = (1, 512) -> shape of each element. List of 128 such elements

		r0 = tf.concat(axis = 0, values = fusion) #Elements in fusion are stacked on each other. (128, 512) 


		#fully connected layer
		predict = tf.add(tf.matmul(r0, w), b, name = "fc1") #(128, 512) * (512, 1595) = (128, 1595) + (1595,) = (128, 1595)

		return r0, predict


	def train_network(self, epoch, filename):

		input_x = tf.placeholder(tf.float32, shape = [self.batch_size, self.group, self.feature_len])# 128 x 5 x 512
		label_x = tf.placeholder(tf.int32, shape = [self.batch_size, self.class_num]) # 128 x 1595

		_, predict = self.create_network(input_x)

		dataset = Data(filename, self.batch_size, self.class_num)
		dataset.load_feature()

		static = tf.equal(tf.argmax(predict, 1), tf.argmax(label_x, 1))
		accuracy = tf.reduce_mean(tf.cast(static, tf.float32)) #Always going to be 1
		tf.summary.scalar("accuracy", accuracy)

		loss = tf.nn.softmax_cross_entropy_with_logits(labels = label_x, logits = predict)
		loss = tf.reduce_mean(loss)
		tf.summary.scalar("loss", loss)

		optim = tf.train.RMSPropOptimizer(learning_rate = 0.001).minimize(loss)

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("log/", sess.graph)

		for i in range(epoch):
			feature_x, labels_x = dataset.next_batch(self.group)

			
			_ = sess.run([optim], feed_dict = {input_x:feature_x, label_x:labels_x})

			if i % 10 == 0:
				_acc, _loss, results = sess.run([accuracy, loss, merged], feed_dict = {input_x:feature_x, label_x: labels_x})
				print("%s\tIteration\t%d\tAccuracy\t%f\tLoss\t%f"%(time.asctime(), i, _acc.item(), _loss.item()))
				writer.add_summary(results, i)

			if i % (epoch / 5) == 0:
				saver.save(sess, "./model/attention.ckpt", global_step = i)



if __name__ == '__main__':
	

	filename = './feature_map_files/feature_map_complete.mat'
	batch_size = 128
	class_num = 1595

	net = Network(batch_size, 512, class_num, 5)
	net.train_network(100000, filename)
