# io = open("C:/Users/Andrew.Bannerman/Desktop/Julia/My_First_Neural_Network/Data/train-images-idx3-ubyte/train-images.idx3-ubyte", n=28)
#write(io)

#open("C:/Users/Andrew.Bannerman/Desktop/Julia/My_First_Neural_Network/Data/train-images-idx3-ubyte/train-images.idx3-ubyte") do file
#    for ln in eachline(file)
#        println("$(length(ln)), $(ln)")
#    end
#end

# Back propogation neural network
# 'Hello World' of neural networks
# Andrew Bannerman 4.21.2019

# Resources
# Cheat Sheet = https://ml-cheatsheet.readthedocs.io/en/latest/forwardpropagation.html
# Matrix multiplication Rules - https://www.mathsisfun.com/algebra/matrix-multiplying.html
# How to get MNIST data as a CSV https://pjreddie.com/projects/mnist-in-csv/
# Kaggle - https://www.kaggle.com/ngbolin/mnist-dataset-digit-recognizer
# mini batch graient descent https://wiseodd.github.io/techblog/2016/06/21/nn-sgd/
# Derivatives expalined: https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/1-%20Neural%20Networks%20and%20Deep%20Learning#gradient-descent-on-m-examples


y = a sin(bx-c) + d
a = 2
period = 3.1416/12



A = 4 * sind(B*(x - 3.1416/2)
x_one = collect(1.0:1.0:10.0)
y = zero(x_one)
for i in 1:length(x_one)
    outie[i] = 2*sind(x_one[i])*3.1416
end
plot(outie)

Events = 71
Time = 15
FREQ = Events/Time
W = 2 .* 3.14 .* FREQ
A = 2

# Back propogation tutorial: https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python

# Needed packages
using DataFrames
using CSV
using Distributions
using Plots
using Random
using Images, CoordinateTransformations, TestImages, OffsetArrays, CoordinateTransformations # For data augmentation
using PaddedViews # for cropping images

#=
Random.seed!(123) # Setting the seed
d = Normal()
x = rand(d, 100)
x = rand(Normal(100, 2), 100)
vars = var(x)
sqrt(vars)
histogram(x)
=#

# Train and test set data preparation

# Load MNIST Data set from .csv format
train_set = CSV.read("C:/Users/Andrew.Bannerman/Desktop/Julia/My_First_Neural_Network/Data/mnist_train.csv",header=false,types=fill(Float64,785))
test_set = CSV.read("C:/Users/Andrew.Bannerman/Desktop/Julia/My_First_Neural_Network/Data/mnist_test.csv",header=false,types=fill(Float64,785))

# Plot Example image
example = Float64.(test_set[2,2:size(test_set,2)])
print(example)
example = rotl90(reshape(example,28,28))
heatmap(example,c = :greys)
print(example)

# Extract labels
train_label = train_set[:,1]
test_label = test_set[:,1]

# Extract Data and exclude the label in the first column
train_set = train_set[:,2:size(train_set,2)]
test_set = test_set[:,2:size(test_set,2)]

# Pre process - Scale the inputs to a range of 0.01 to 1.00
m_train = Array{Float64}(zeros(nrow(train_set),size(train_set,2))) # empty matrix
@inbounds for i in 1:nrow(train_set)
    @inbounds for j in 1:size(train_set,2)
    m_train[i,j] = (train_set[i,j] / 255.0 * 0.99) + 0.01
    end
end

m_test = Array{Float64}(zeros(nrow(test_set),size(test_set,2))) # empty matrix
@inbounds for i in 1:nrow(test_set)
    @inbounds for j in 1:size(test_set,2)
    m_test[i,j] = (test_set[i,j] / 255.0 * 0.99) + 0.01
    end
end

# Convert matrix to DataFrame and append the labels back to the train and test set
scaled_orig_train_inputs = hcat(train_label,m_train)
scaled_test_inputs = hcat(test_label,m_test)

# Plot the new scale example
example = Float64.(scaled_orig_train_inputs[1,2:size(scaled_orig_train_inputs,2)])
example = rotl90(reshape(example,28,28))
heatmap(example, c = :greys)

#---------
# Data Augmentation
#---------

# Rotate images
training_set_aug_minus = zeros(size(scaled_orig_train_inputs,1),size(scaled_orig_train_inputs,2))
training_set_aug_plus = zeros(size(scaled_orig_train_inputs,1),size(scaled_orig_train_inputs,2))
degree_rot_minus = -.3
degree_rot_plus = .3
i=1
for i in 1:size(scaled_orig_train_inputs,1)
        label = scaled_orig_train_inputs[i,1]
        example = Float64.(scaled_orig_train_inputs[i,2:size(scaled_orig_train_inputs,2)])
        example = reshape(example,28,28)
        #rot_minus = LinearMap(RotMatrix(degree_rot_minus))
        #rot_plus = LinearMap(RotMatrix(degree_rot_plus))
        #imgw_minus = warp(example_minus, rot_minus, axes(example_plus)) #  axes(example_minus) = crops
        #imgw_plus = warp(example_plus, rot_plus, axes(example_plus)) # axes() = crops
        imgw_minus = imrotate(example, degree_rot_minus, axes(example))
        imgw_plus = imrotate(example, degree_rot_plus, axes(example))
        imgw_minus = reshape(imgw_minus,1,784)
        imgw_plus = reshape(imgw_plus,1,784)
            for k in 1:size(imgw_minus,2)
                if isnan(imgw_minus[k]) # Convert NaN to 0.0
                    imgw_minus[k] = 0.01
                end
                if isnan(imgw_plus[k]) # Convert NaN to 0.0
                    imgw_plus[k] = 0.01
                end
            end
            #imgw_plus .= (imgw_plus ./ 255.0 .* 0.99) .+ 0.01 # Scale the outputs
            #imgw_minus .= (imgw_minus ./ 255.0 .* 0.99) .+ 0.01 # Scale the outputs
            training_set_aug_minus[i,1] = label
            training_set_aug_minus[i,2:size(scaled_orig_train_inputs,2)] = imgw_minus
            training_set_aug_plus[i,1] = label
            training_set_aug_plus[i,2:size(scaled_orig_train_inputs,2)] = imgw_plus
        print("This is iteration i",i,"\n")
end

    maximum(imgw_plus)


# Combine augmented data with the original
# note may want to add shuffling?
scaled_train_inputs = vcat(scaled_orig_train_inputs,training_set_aug_minus,training_set_aug_plus)
scaled_train_inputs = scaled_orig_train_inputs

# Shuffle the augmented + original data
scaled_train_inputs = scaled_train_inputs[shuffle(1:end), :]

#---------
# Activation Functions
#---------

function sigmoid(x)
    return 1 / (1.0 + exp(-x))
end

function sigmoid_derivative(x)
    s = 1 / (1 + exp(-x))
    return s * (1 - s)
end

function ReLU(x)
    #return x*(x>0)
    return maximum([0.0,x])
end

function ReLU_derivative(x)
    if x <= 0
    end
    return 0
end

"""
Leaky ReLU
f(x)=1(x<0)(αx)+1(x>=0)(x) where α = some constant
"""
function ReLuL(x)
    if (x > 0.0)
        return x
    elseif (x <= 0.0)
        return x * 0.01
    end
end

 function ReLUl_derivative(x, alpha=0.01)
      if x <= 0
      end
      return alpha
  end

  # https://deepnotes.io/softmax-crossentropy

"""
softmax - final result should sum to 1
"""
  function softmax(x)
  e_x = exp.(x .- maximum(x))
return e_x / sum(e_x) # only difference
end

""" Softmax derivative
"""
function delta_cross_entropy(A2::Array{Float64,2}, Y::Array{Float64,2})
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = size(Y)[1]
    grad = softmax(A2)
    network_output = argmax(A2)[1]
    grad[network_output] -= 1
    #grad = grad./m
    return grad
end

grad[range(m),y] -= 1

y = [0.0,4.0,0.0]
testing = [1.0,4.0,3.0]
grad = softmax(testing)

grad[1:length(testing),y]

sum(testing, dims = 1)

maximum(testing)
 e_x = exp.(testing .- maximum(testing))
#---------
# Layers
# https://medium.freecodecamp.org/building-a-3-layer-neural-network-from-scratch-99239c4af5d3
#---------

function dropout(d_in::Array{Float64,2}; p::Float64=.8)
    out = (d_in .* rand(Binomial(1,p), size(d_in)[1])) ./ p
return out
end

activation = "sigmoid"
drop_out = false
function forward_prop(W1::Array{Float64,2}, b1::Array{Float64,2}, W2::Array{Float64,2}, b2::Array{Float64,2}, X::Array{Float64,2}, Y::Array{Float64,2}; drop_out::Bool=true, p::Float64=.5, activation::String="sigmoid", softmax_bool::Bool=true)
    Z1 = W1 * X + b1
    if drop_out == false
        if activation == "sigmoid"
    A1 = sigmoid.(Z1)
elseif activation == "relu"
    A1 = ReLU.(Z1)
elseif activation == "leaky_relu"
        A1 = ReLuL.(Z1)
    end
    elseif drop_out == true
        if activation == "sigmoid"
    A1 = sigmoid.(Z1)
    A1 = dropout(A1,p=.8)
    elseif activation == "relu"
    A1 = ReLU.(Z1)
    A1 = dropout(A1,p=.8) # Apply drop out by hidden output * 0,1's / p (1/p - scaled so we dont need to apply at test time)(inverted dropout)
elseif activation == "leaky_relu"
    A1 = ReLuL.(Z1)
    A1 = dropout(A1,p=.8) # Apply drop out by hidden output * 0,1's / p (1/p - scaled so we dont need to apply at test time)(inverted dropout)
        end
    end
    Z2 = (W2 * A1) + b2
    if activation == "sigmoid" && softmax_bool == false
    A2 = sigmoid.(Z2)
elseif activation == "relu" && softmax_bool == false
    A2 = ReLU.(Z2)
elseif activation == "leaky_relu" && softmax_bool == false
    A2 = ReLuL.(Z2)
elseif softmax_bool == true
    A2 = softmax(Z2)
    end
    return Z1, A1, Z2, A2, b1, b2
end

#----
# back prop math: http://deeplizard.com/learn/video/G5b4jRBKNxw
# https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/1-%20Neural%20Networks%20and%20Deep%20Learning#gradient-descent-on-m-examples
# numpy - https://github.com/sprtkd/scratchNN/blob/master/basicNNtrain.py
# to do: https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%202/
# full with momentum , batches https://zhenye-na.github.io/2018/09/09/build-neural-network-with-mnist-from-scratch.html
# https://github.com/sprtkd/scratchNN/blob/master/basicNNtrain.py
#----
function backward_prop(W1::Array{Float64,2}, b1::Array{Float64,2}, W2::Array{Float64,2}, b2::Array{Float64,2}, X::Array{Float64,2}, Y::Array{Float64,2}, Z1::Array{Float64,2}, A1::Array{Float64,2}, Z2::Array{Float64,2}, A2::Array{Float64,2}; learning_rate::Float64=0.1,activation::String="relu")
    # error at last layer
    dZ2 = A2 - Y
    dW2 = (dZ2 * A1')
    db2 = sum(dZ2, dims = 2)
    # back propgate through first layer
    if activation == "sigmoid"
        dZ1 = (W2' * dZ2) .* sigmoid_derivative.(A1) #(1 .- A1.^2)
    elseif activation == "relu"
        dZ1 = (W2' * dZ2) .* ReLU_derivative.(A1) #(1 .- A1.^2)
    elseif activation == "leaky_relu"
        dZ1 = (W2' * dZ2) .* ReLUl_derivative.(A1) #(1 .- A1.^2)
    end
    # gradients at first layer
    dW1 = (dZ1 * X')
    db1 = sum(dZ1,dims = 2)
   # Update params
    W1 = W1 .- (learning_rate * dW1)
     b1 = b1 .- (learning_rate * db1)
     W2 = W2 .- (learning_rate * dW2)
     b2 = b2 .- (learning_rate * db2)
    # print("check w1 is updating",W2[1])

    return  dW1, db1, dW2, db2, W1, b1, W2, b2
end

#---------
# Cost Function
#---------

function cost_function(A2::Array{Float64,2}, Y::Array{Float64,2}; activation::String="sigmoid")
    if activation == "sigmoid"
    m = size(Y)[2]
    #cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    cost = (-1/m) * sum((Y' * log.(A2)) .+ ((1 .- Y')* log.(1 .- A2)))
        #cost = (-1/m) * sum((Y * log.(A2')) .+ ((1 .- Y)* log.(1 .- A2')))
elseif activation == "relu" || activation == "leaky_relu"
    return Inf
end
return cost
end

function cost_function(A2::Array{Float64,2}, Y::Array{Float64,2})
    L_sum = sum(-A2' * log.(A2))
return L_sum
end

#---------
# Initialize network
#---------

# Define neural network
# Number of input, hidden and output nodes
input_nodes = size(train_set,2) # equal to the size of the 28 * 28 image (784 elements)
hidden_nodes = 200
output_nodes = 10 # Equal to the number of labels in this case numbers 0 to 9 (10 elements total)

# Create instance of neural network
inodes = input_nodes
hnodes = hidden_nodes
onodes = output_nodes
# Weight initialization
# w = np.random.randn(n) * sqrt(2.0/n) for relu
# w = np.random.randn(n) / sqrt(n) # other
#W1 = randn(hnodes,inodes) * inodes^(-0.5)
#W2 = randn(onodes,hnodes) * hnodes^(-0.5)

W1 = randn(hnodes,inodes) * 0.01
W2 = randn(onodes,hnodes) * 0.01

#W1 = randn(hnodes,inodes) * sqrt(2.0/inodes)
#W2 = randn(onodes,hnodes) * sqrt(2.0/hnodes)
# Initializing the biases as 0's
b1 = zeros(hnodes,1)
b2 = zeros(onodes,1)

epochs = 5
# Check that the neural network is converging
# Initialize training progress
learning_curve = fill(0,size(scaled_train_inputs,1))
training_accuracy = fill(0.0,size(scaled_train_inputs,1))
i = 1
j = 1
@inbounds for i in 1:epochs
    # go through all records in the training data set
    @inbounds for j in 1:size(scaled_train_inputs,1)
        # Forward propogation
        label = Int64.(scaled_train_inputs[j,1])
        X = reshape(Float64.(scaled_train_inputs[j,2:size(scaled_train_inputs,2)]),784,1)
        # create the target output values (all 0.01, except the desired label which is 0.99)
        Y = zeros(10,1)
        Y .= 0.01 .+ Y
        # Set the target .99 to the correct index position within the target array
        Y[label+1] = 0.99 # We +1 because the number convention starts at 0 to 9. Thus array position 6 is actually target number 5.
global Z1, A1, Z2, A2, b1, b2 = forward_prop(W1, b1, W2, b2, X, Y; drop_out=false, p=.5,activation="sigmoid",softmax_bool=true)
cost_rate = cost_function(A2,Y)
global dW1, db1, dW2, db2, W1, b1, W2, b2 = backward_prop(W1, b1, W2, b2, X, Y, Z1, A1, Z2, A2; learning_rate=.1,activation="sigmoid")
#print("check training weight being update", dW1[1],"\n")
network_train_output = argmax(A2)[1]-1 # adjust index position to account from 0 start
if (label == network_train_output)
    # network's answer matches correct answer, add 1 to learning curve
    learning_curve[j] = 1
else
    # network's answer doesn't match correct answer, add 0 to learning curve
    learning_curve[j] = 0
end
if (j == 1)
    global learning_curve = fill(0,size(scaled_train_inputs,1))
end
print("Cost after iteration ",j," ",cost_rate,"\n")
print("Training Accuracy ",(sum(learning_curve) / j),"\n")
print("This is epoch ",i,"\nThis is iteration ",j,"\n")

    end
end

sum(learning_curve) / 60000

print(learning_curve)

# Plot the learning curve
plot(cumsum(learning_curve),title="Learning Curve")

#---------
# Test Function
#---------

# Function for applying the network to unseen or test data not seen (trained) by the network before
function test_nn(W1::Array{Float64,2}, W2::Array{Float64,2},X::Array{Float64,2})
    Z1 = W1 * X
        if activation == "sigmoid"
    A1 = sigmoid.(Z1)
elseif activation == "relu"
    A1 = ReLU.(Z1)
    end
    Z2 = (W2 * A1)
    if activation == "sigmoid"
    A2 = sigmoid.(Z2)
elseif activation == "relu"
    A2 = ReLU.(Z2)
    end
    return A2
end

#---------
# Test Network on Individual image
#---------

# Call the network on an indivdual entry
example_call = Float64.(scaled_test_inputs[500,2:size(scaled_test_inputs,2)])
example_call = reshape(example_call,784,1)
example_call_image = rotl90(reshape(example_call,28,28))
# Plot Number
heatmap(example_call_image,c = :greys)
# Grab the known correct label
correct_label = scaled_test_inputs[500,1]
# Apply the trained network to the input example
outputs = test_nn(W1, W2, example_call)
# Check what the label thought the number was
network_label = argmax(outputs)[1]-1
if network_label ==  correct_label
    print("The neural network correctly identified the hand written number - ", network_label)
else
    print("The neural network incorrectly identified the hand written number")
end

print(example_call_image)

#---------
# Test Neural Network on Test Data
#---------

# Initialize the scorecard output the same size as the number of test set labels
scorecard = fill(0,nrow(scaled_test_inputs))
test_accuracy = fill(0.0,nrow(scaled_test_inputs))

# Input the test set data over the trained neural network to see how well it does
i=1
@inbounds for i in 1:nrow(scaled_test_inputs)
    correct_label = Int64.(scaled_test_inputs[i,1])
    inputs = reshape(Float64.(scaled_test_inputs[i,2:size(scaled_test_inputs,2)]),784,1) # rotr90() for changing dim to [784,1] from [1,784] to meet matrix multiplication rules
    # query the network
    outputs = test_nn(W1, W2, b1,b2,inputs) # Test the network over the test set data (wih and who already set during training - no weight updating happens this time round)
    # Find element position of what the network thinks the output is ie if output is a .96 in element position 6 then the network thinks the label was 5.
    # This is because we have hand written numbers 0 to 9, total 10 outputs. Starting from index position 1 = label 0 hence index position 6 then = label 5
    label = argmax(outputs)[1]-1 # adjust index position to account from 0 start
    # append correct or incorrect to list
    if (label == correct_label)
        # network's answer matches correct answer, add 1 to scorecard
        scorecard[i] = 1
    else
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard[i] = 0
        end
        print("This is iteration ", i,"\n")
        print("Accuracy ",sum(scorecard) / i,"\n")
        test_accuracy[i] = sum(scorecard) / i
end

# Percentage Correct
percentage = sum(scorecard) / size(scorecard,1)

# Change log
# 5.12.2019 - Changed the back propogation derivatives to allow different activation functions
# 5.12.2019 - Added bias
# 4.23.2019 - Added first implemntation of drop out
# 4.22.2019 - added a % print function to check train / test accuracy
