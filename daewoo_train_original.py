import pandas as pd
import os
from tensorflow.contrib.data import Dataset
from daewoo_module import *
import time
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


root_dir = ".\\input_data"
# img_dir = os.path.join(root_dir, '/figure'). 이미지는 os.path.join쓰면 안되는경우가 있음
img_dir = "./input_data/figure/"
logs_path = os.path.join(root_dir, "graph")

img = np.array([img_dir + x for x in os.listdir(img_dir)])
label = pd.read_csv(os.path.join(root_dir, 'description.csv'), engine='python')

# Classification과 regression 선택
classification = True

batch_size = 64
epochs = 1

if classification is True:
	label = pd.cut(label['WVHT ft.y'], bins=[0, 5, 10, 100], labels=[0, 1, 2], include_lowest=True)
	label = np.array(label)
else:
	label = label['WVHT ft.y'].values
	label = ((label - np.mean(label)) / np.std(label)).reshape(-1, 1)


train_img_tensor, train_label_tensor, test_img_tensor, test_label_tensor = set_input(img, label)


train_imgs = Dataset.from_tensor_slices((train_img_tensor, train_label_tensor))
test_imgs = Dataset.from_tensor_slices((test_img_tensor, test_label_tensor))
infer_imgs = Dataset.from_tensor_slices((test_img_tensor, test_label_tensor))

if classification is True:
	train_imgs = train_imgs.map(input_tensor).batch(batch_size).shuffle(buffer_size=100).repeat()
	test_imgs = test_imgs.map(input_tensor).batch(batch_size).shuffle(buffer_size=100).repeat()
	infer_imgs = infer_imgs.map(input_tensor).batch(int(test_label_tensor.shape[0]))
else:
	train_imgs = train_imgs.map(input_tensor_regression).batch(batch_size).shuffle(buffer_size=100).repeat()
	test_imgs = test_imgs.map(input_tensor_regression).batch(batch_size).shuffle(buffer_size=100).repeat()
	infer_imgs = infer_imgs.map(input_tensor).batch(int(test_label_tensor.shape[0]))

train_iterator = train_imgs.make_initializable_iterator()
test_iterator = test_imgs.make_initializable_iterator()
infer_iterator = infer_imgs.make_initializable_iterator()
handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(handle, train_imgs.output_types, train_imgs.output_shapes)
x, y = iterator.get_next()


train_batches = round(0.8 * len(label)) // batch_size
test_batches = (len(label) - round(0.8 * len(label))) // batch_size


model = VGG16(x, y, bn=True, classification=classification)

if classification is True:
	model_name = "VGG16_classification_crop"
else:
	model_name = "VGG16_regression_crop"

start_time = time.time()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


sess = tf.Session(config=config)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
train_handle = sess.run(train_iterator.string_handle())
test_handle = sess.run(test_iterator.string_handle())
# infer_handle = sess.run(infer_iterator.string_handle())
train_writer = tf.summary.FileWriter(os.path.join(logs_path, model_name, 'train'), sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(logs_path, model_name, 'test'))

LEARNING_RATE = 0.001
optimizer = model.rms

if classification is True:

	print("Training!")
	for i in range(epochs):
		print("-------{} Epoch--------".format(i + 1))
		sess.run(train_iterator.initializer)
		sess.run(test_iterator.initializer)
		for j in range(train_batches):
			summary, _, acc, loss_ = sess.run([model.merged_summary_op, optimizer, model.accuracy, model.loss],
			                                  feed_dict={handle: train_handle, model.learning_rate: LEARNING_RATE})
			step = tf.train.global_step(sess, model.global_step)
			print("Training Iter : {}, Acc : {}, Loss : {:.4f}".format(step, acc, loss_))

			if j % 10 == 0:
				train_writer.add_summary(summary, step)
				summary, acc, loss_ = sess.run([model.merged_summary_op, model.accuracy, model.loss],
				                               feed_dict={handle: test_handle})
				print("Validation Iter : {}, Acc : {}, Loss : {:.4f}".format(step, acc, loss_))
				test_writer.add_summary(summary, step)

	print("-----------End of training-------------")

	end_time = time.time() - start_time
	print("{} seconds".format(end_time))

	saver.save(sess, os.path.join(logs_path, 'VGG16_classification_crop', model_name))

else:
	print("Training!")
	for i in range(epochs):
		print("-------{} Epoch--------".format(i + 1))
		sess.run(train_iterator.initializer)
		sess.run(test_iterator.initializer)
		for j in range(train_batches):
			summary, _, loss_ = sess.run([model.merged_summary_op, optimizer, model.loss],
			                             feed_dict={handle: train_handle, model.learning_rate: LEARNING_RATE})
			step = tf.train.global_step(sess, model.global_step)
			print("Training Iter : {}, Loss : {:.4f}".format(step, loss_))

			if j % 10 == 0:
				train_writer.add_summary(summary, step)
				summary, loss_ = sess.run([model.merged_summary_op, model.loss],
				                          feed_dict={handle: test_handle})
				print("Validation Iter : {}, Loss : {:.4f}".format(step, loss_))
				test_writer.add_summary(summary, step)

	print("-----------End of training-------------")

	end_time = time.time() - start_time
	print("{} seconds".format(end_time))

	saver.save(sess, os.path.join(logs_path, 'VGG16_regression_crop', model_name))


# Inference
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver = tf.train.import_meta_graph(os.path.join(logs_path, 'VGG16_classification_crop', 'VGG16_classification_crop.meta'))
saver.restore(sess, tf.train.latest_checkpoint(os.path.join(logs_path, 'VGG16_classification_crop')))
sess.run(tf.global_variables_initializer())


infer_imgs = infer_imgs.map(input_tensor).batch(int(test_label_tensor.shape[0]))
infer_handle = sess.run(infer_iterator.string_handle())
sess.run(infer_iterator.initializer)
prob = sess.run(model.y_prob, feed_dict={handle: infer_handle})
