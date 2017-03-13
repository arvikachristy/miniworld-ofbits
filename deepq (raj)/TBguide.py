# Implementing Tensorboard to your Code

# 1. You will need to create a Tensorflow “Summary Writer” that writes values to the Tensorboard. 
# You will need to specify a directory as a variable that will save an event file to the directory 
# This file adds summaries and logs data to it. 
# Assume 'summaries_dir' is a variable with the path 
# The below code can be added to the initialization of your network 

summary_writer = None 
self._build_model() 
if summaries_dir:  #If you have specified a directory for 'summaries_dir'
	summaries_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
	if not os.path.exists(summaries_dir):
		os.makedirs(summaries_dir) #If summaries_dir does not exist
	self.summary_writer = tf.summary.FileWriter(summaries_dir) #FileWrite is the summary writer, and we specify where we should write the event file to

# 2. Now we can write variables to your Tensorboard 
# IMPORTANT : There are different processes if you want to write a regular Python variable or a Tensor

# 2. a) 1. If you want to write a Tensor to the Tensorboar, you can use these functions depending how you want to represent them in the Tensorboard:
	tf.summary.scalar(name = "Name of Variable", tensor = TheActualTensor) #Scalar Graph
	tf.summary.histogram (name = "Name of Variable", tensor = TheActualTensor) #Histogram 
	tf.summary.image (name = "Name of Variable", tensor = TheActualTensor) #Image (useful if you want to check pixels)
	tf.summary.audio (name = "Name of Variable", tensor = TheActualTensor) #Audio (don't think we'll need it)

	#For example, in my code :
	 self.summaries = tf.summary.merge([
	      tf.summary.scalar("loss",self.loss),
	      tf.summary.histogram("loss_hist",self.losses),
	      tf.summary.histogram("q_values_hist",self.predictions),
	      tf.summary.scalar("max_q_value",tf.reduce_max(self.predictions))])

	#tf.summary.merge lets you add multiple tensors easily
	#2. Once you gather your tensors, write them to the tensorboard
	self.summary_writer.add_summary(summaries, global_step)

	#global_step is tf.contrib.framework.get_global_step()

#2. b) 1. If you want to write regular Python variables to the Tensorboard, first create a Summary() wrapper

	episode_summary = tf.Summary()

	#2. Then add variables to the Summary wrapper
	episode_summary.value.add(simple_value = TheVariable, tag = "Name to display in Tensorboard")
	#You can further add variables to this one Summary wrapper

	#3. When done, add your variables to your Summary Writer
	qnetwork.summary_writer.add_summary(episode_summary, xAxisVariable)

	#xAxisVariable could be the current number of total steps at that point of time
	#4. Add flush to make sure it writes to disk
	qnetwork.summary_writer.flush()

	#Here's an example using I used in my code :
	episode_summary = tf.Summary()
    	episode_summary.value.add(simple_value = stats.episode_rewards[i], node_name = "episode_reward", tag = "episode_reward") #Plots episode reward against time steps
    	episode_summary.value.add(simple_value = stats.episode_lengths[i], node_name = "episode_lengths", tag = "episode_lengths") #Plots episode length against time steps 
    	qnetwork.summary_writer.add_summary(episode_summary, total_t) #total_t is the current total time step at the point of writing these summaries
    	qnetwork.summary_writer.flush()

#3. Once you have your variables ready, run your agent, and then run this line on your terminal :

"tensorboard --logdir = path/to/log-directory"

#This will run the Tensorboard local server, and will provide you an address
#Run this address in your browser to load the Tensorboard 
