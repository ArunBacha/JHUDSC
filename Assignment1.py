

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

KNeighbours = 9
Errors = 0

HouseNumbers = keras.datasets.mnist

(TrainImages, TrainClass), (TestImages, TestClass) = HouseNumbers.load_data()


X = tf.placeholder(TrainImages.dtype, shape=TrainImages.shape)
Y = tf.placeholder(TestImages.dtype, shape=TestImages.shape[1:])
xThresholded = tf.cast(X, tf.float32) 
yThresholded = tf.cast(Y, tf.float32)
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xThresholded, yThresholded)), axis=[1,2]))
findKClosestTrImages = tf.contrib.framework.argsort(distance, direction='ASCENDING') # sorting (image) indices in order of ascending metrics, pick first k in the next step
findLabelsKClosestTrImages = tf.gather(TrainClass, findKClosestTrImages[0:KNeighbours]) # doing trLabels[findKClosestTrImages[0:paramk]] throws error, hence this workaround
findULabels, findIdex, findCounts = tf.unique_with_counts(findLabelsKClosestTrImages) # examine labels of k closest Train images
findPredictedLabel = tf.gather(findULabels, tf.argmax(findCounts)) # assign label to test image based on most occurring labels among k closest Train images

# Let's run the graph
TestImagesNum = np.shape(TestClass)[0]
TrainImagesNum = np.shape(TrainClass)[0] # so many train images

with tf.Session() as sess:
  for iTeI in range(0,TestImagesNum): # iterate each image in test set
    predictedLabel = sess.run([findPredictedLabel], feed_dict={X:TrainImages, Y:TestImages[iTeI]})   

    if predictedLabel != TestClass[iTeI]:
      Errors += 1
      print(Errors,"/",iTeI)
      print("\t\t", predictedLabel[0], "\t\t\t\t", TestClass[iTeI])
      
      if (1):
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(TestImages[iTeI])
        plt.title('Test Image has label %i but predicted as %i' %(TestClass[iTeI],predictedLabel[0]))
        plt.show()
      
print("Classification errors = ",Errors)
print("Test Image Numbers = ", TestImagesNum  )
print("Accuracy of the model = ", 100*(TestImagesNum - Errors)/TestImagesNum)