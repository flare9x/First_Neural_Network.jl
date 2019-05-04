# Neural Network (MNIST) Hand Written Numbers

Julia implementation of a back propogation neural network as inspired from the book: Make your own neural network. 

Initial score 0.967428 accuracy on the Kaggle MNIST Digit Recognizer data set. That is training on 
42000 hand written numbers and the test set consisting of 28000 hand written numbers. 

# Improving The Network Accuracy 
Added data augmentation using image rotations (-/+) (increased the training images by a factor of 3x)
Decreased the learning rate 

With the above the standard neural network achieved 0.97914 accuracy on the Kaggle MNIST Digit Recognizer data set. That is training on 
126000 hand written numbers and the test set consisting of 28000 hand written numbers. 

#to do 
Different activation functions 
Different loss functions 
