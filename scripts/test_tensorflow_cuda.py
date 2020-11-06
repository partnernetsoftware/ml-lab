import tensorflow as tf

#test = tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None)
#test = tf.test.is_gpu_available( cuda_only=False )
#test = tf.test.is_gpu_available( cuda_only=True )
#print(test)
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print( tf.config.list_physical_devices('GPU'))
