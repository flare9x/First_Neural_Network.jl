#io = open("C:/Users/Andrew.Bannerman/Desktop/Julia/My_First_Neural_Network/Data/train-images-idx3-ubyte/train-images.idx3-ubyte", n=28)
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

# Needed packages
using DataFrames
using CSV
using Distributions
using Plots
using Random

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
example = convert(Array{Float64}, test_set[2,2:length(test_set)])
example = rotl90(reshape(example,28,28))
heatmap(example)

# Extract labels
train_label = train_set[:,1]
test_label = test_set[:,1]

# Extract Data and exclude the label in the first column
train_set = train_set[:,2:length(train_set)]
test_set = test_set[:,2:length(test_set)]

# Pre process - Scale the inputs to a range of 0.01 to 1.00
m_train = Array{Float64}(zeros(nrow(train_set),length(train_set))) # empty matrix
@inbounds for i in 1:nrow(train_set)
    @inbounds for j in 1:length(train_set)
    m_train[i,j] = (train_set[i,j] / 255.0 * 0.99) + 0.01
    end
end

m_test = Array{Float64}(zeros(nrow(test_set),length(test_set))) # empty matrix
@inbounds for i in 1:nrow(test_set)
    @inbounds for j in 1:length(test_set)
    m_test[i,j] = (test_set[i,j] / 255.0 * 0.99) + 0.01
    end
end

# Convert matrix to DataFrame and append the labels back to the train and test set
scaled_train_inputs = DataFrame(hcat(train_label,m_train))
scaled_test_inputs = DataFrame(hcat(test_label,m_test))

# Plot the new scale example
example = convert(Array{Float64}, scaled_train_inputs[1,2:length(scaled_train_inputs)])
example = rotl90(reshape(example,28,28))
heatmap(example)

# Build the back propogation neural network

# Create sigmoid activation function
function sigmoid(x)
    return 1 / (1.0 + exp(-x))
end

# Create the training function
# Step 1 - feed the data through the 3 layers whilst adjusting the inputs by link weights + squishing with sigmoid function at each output node.
# Layer 1 = inputs (inputs * weights)
# Layer 2 = hidden layer = squish with sigmoid the output of layer 1 - the output of hidden layer then is * by link weights (who) which then becomes the input to the output layer
# Layer 3 = output = squish output of hidden layer by sigmoid to obtain final output
# Step 2
# Calculate the error - the network output (what the network thinks the answer is) - the target
# Step 3 - Update the link weights in proporation to the error (ie if a link weight is higher than another link weight to same node - adjust the larger weight by the bigger fraciton)
# Output Errors = targets - final_outputs
# Hidden layer output errors = who' * output_errors
# Using chain rule - update the link weights from the hidden layer to output (who) and again for the input to the hidden layer (wih)
# This updating of weights is how the network learns!
function train(wih::Array{Float64,2}, who::Array{Float64,2},inputs::Array{Float64,2}, targets::Array{Float64,2};  lr::Float64=0.1) #wih, who, lr, inputs, targets)
    # Matrix multiplication Rules - https://www.mathsisfun.com/algebra/matrix-multiplying.html
    # [m * n] * [n * p] = [m * p] # the last n must match the first n
    # [row, column]
    # wih dim = [200,784]
    # input dim = [784, 1]
    hidden_inputs = wih * inputs # hidd_input dim = [200, 1]

    # Apply sigmoid function to each element the hidden layer output
    # hidden_inputs dim = [200, 1]
    hidden_outputs = sigmoid.(hidden_inputs)

    # Apply the weights to the hidden layer outputs which then are the final layers inputs
    # who dims = [10,200]
    # hidden_outputs dims = [200, 1]
    final_inputs = who * hidden_outputs

    # Apply sigmoid function to each element the final layer output
    # final_inputs dim = [10,1]
    final_outputs = sigmoid.(final_inputs)

    # output layer error is the (target - actual)
    # output_errors dim = [10,1]
    output_errors = targets - final_outputs

    # hidden_errors dim = [200,1]
    # Find error between target and output
    hidden_errors = who' * output_errors

    # Update weights of the neural network - this is how the network learns by adjusting the link weights in propotation to the link weight and error.
    # update the weights for the links between the hidden and output layers
    who .= who .+ (lr .* (output_errors .* final_outputs .* (1.0 .- final_outputs)) * hidden_outputs')

    # update the weights for the links between the input and hidden layers
    wih .= wih .+ (lr .* (hidden_errors .* hidden_outputs .* (1.0 .- hidden_outputs)) * inputs')

    return(wih, who, final_outputs)
end

#=
# Example of link weight updating (prior to using calclus chain rule)
# Refine weight on output node (1x node example with two w11 and w21 connected links)
# Splits the error in proportion to the error ie if w11 is contributing more to error, adjust that by bigger weight.
# Fraction of e1 to refine w11
w11 = 6.0  # weight 1, 1 means node connection 1 to node 1
w21 = 3.0 # weight 2, 1 means node connection 2 to 1
refine_w11 = w11 / (w11+w21) # 2.3

# Fraction of e1 to refine w21
refine_w11 = w21 / (w21+w11) # 1/3

# Recombine split errors for the links using the error backpropogation
# ehidden,1 = sum of split errors on links w11 and w12
# Link weights beween hidden and output layer example
w11 = 2.0
w12 = 1.0
w21 = 3.0
w22 = 4.0
e1 = 0.8 # final output error
e2 = 0.5 # final output error
# Find the sum of the errors on e1 and e2
error_out_e1 = e1 * (w11 / (w11 + w21)) + e2 * (w12 / (w12 + w22)) # Sum of split errors
# Flip to find the other summed link error
error_out_e2 = e2 * (w22 / (w22 + w12)) + e1 * (w21 / (w21 + w11)) # Sum of split errors

# Back propogate error_out_e1 and error_out_e2 to the hidden layer input weights
w11 = 3.0
w12 = 1.0
w21 = 2.0
w22 = 7.0
e1 = error_out_e1 # Hidden layer output sum of proportional errors
e2 = error_out_e2 # Hidden layer output sum of proportional errors
error_out_e1 = e1 * (w11 / (w11 + w21)) + e2 * (w12 / (w12 + w22)) # Sum of split errors
error_out_e2 = e2 * (w22 / (w22 + w12)) + e1 * (w21 / (w21 + w11))

# simply in matrix
(w11 / (w11 + w21)) (w12 / (w12 + w22))     e1
(w21 / (w21 + w11)) (w22 / (w22 + w12))     e2
=#

# Function for applying the network to unseen or test data not seen (trained) by the network before
function query(wih::Array{Float64,2}, who::Array{Float64,2},inputs::Array{Float64,2})
    # calculate signals into hidden layer
    hidden_inputs = wih * inputs
    # calculate the signals emerging from hidden layer
    hidden_outputs = sigmoid.(hidden_inputs)

    # calculate signals into final output layer
    final_inputs = who * hidden_outputs
    # calculate the signals emerging from final output layer
    final_outputs = sigmoid.(final_inputs)

    return(final_outputs)
end

# Number of input, hidden and output nodes
input_nodes = length(train_set) # equal to the size of the 28 * 28 image (784 elements)
hidden_nodes = 200
output_nodes = 10 # Equal to the number of labels in this case numbers 0 to 9 (10 elements total)

# learning rate
learning_rate = 0.01

# Create instance of neural network
inodes = input_nodes
hnodes = hidden_nodes
onodes = output_nodes
# As randn generates numbers from a standard normal distribution we have to multiply by the standard deviation
wih = randn(hnodes,inodes) * inodes^(-0.5)
who = randn(onodes,hnodes) * hnodes^(-0.5)

# Train the neural network
# 4.22.2019 - Add functinality to save the numbers which the network did not get correct
# Epochs is the number of times the training data set is used for training
epochs = 20
# Initialize training progress
learning_curve = fill(0,nrow(scaled_train_inputs))
# Check that the neural network is converging
j =1
@inbounds for i in 1:epochs
    # go through all records in the training data set
    @inbounds for j in 1:nrow(scaled_train_inputs)
        # Subset by row [j,1] = label [j,2:length(scaled_train_inputs)] is the scaled image data
        label = Int64.(scaled_train_inputs[j,1])
        inputs = rotr90(convert(Array{Float64},scaled_train_inputs[j,2:length(scaled_train_inputs)])) # rotr90() for changing dim to [784,1] from [1,784] to meet matrix multiplication rules
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = zeros(10,1)
        targets .= 0.01 .+ targets
        # Set the target .99 to the correct index position within the target array
        targets[label+1] = 0.99 # We +1 because the number convention starts at 0 to 9. Thus array position 6 is actually target number 5.
        global wih, who, final_outputs = train(wih, who, inputs, targets; lr=learning_rate)
        # Check what the network thought the number was
        network_train_output = argmax(final_outputs)[1]-1 # adjust index position to account from 0 start
        if (label == network_train_output)
            # network's answer matches correct answer, add 1 to learning curve
            learning_curve[j] = 1
        else
            # network's answer doesn't match correct answer, add 0 to learning curve
            learning_curve[j] = 0
        end
        if (j == 1)
            global learning_curve = fill(0,nrow(scaled_train_inputs))
        end
        print("This is epoch ",i,"\nThis is iteration ",j,"\nCheck hidden layer output weight links are updating ", who[1],"\n")
        print("Training Accuracy ",(sum(learning_curve) / j),"\n")
    end
end

sum(learning_curve) / 60000

print(learning_curve)

# Plot the learning curve
plot(cumsum(learning_curve),title="Learning Curve")

# Call the network on an indivdual entry
example_call = inputs = rotr90(convert(Array{Float64},scaled_test_inputs[500,2:length(scaled_test_inputs)]))
example_call_image = rotl90(reshape(example_call,28,28))
# Plot Number
heatmap(example_call_image)
# Grab the known correct label
correct_label = scaled_test_inputs[500,1]
# Apply the trained network to the input example
outputs = query(wih, who, example_call)
# Check what the label thought the number was
network_label = argmax(outputs)[1]-1
if network_label ==  correct_label
    print("The neural network correctly identified the hand written number - ", network_label)
else
    print("The neural network incorrectly identified the hand written number")
end

print(example_call_image)

# Test the neural network
# Initialize the scorecard output the same size as the number of test set labels
scorecard = fill(0,nrow(scaled_test_inputs))

# Input the test set data over the trained neural network to see how well it does
i=1
@inbounds for i in 1:nrow(scaled_test_inputs)
    let wih = wih, who = who
    correct_label = Int64.(scaled_test_inputs[i,1])
    inputs = rotr90(convert(Array{Float64},scaled_test_inputs[i,2:length(scaled_test_inputs)])) # rotr90() for changing dim to [784,1] from [1,784] to meet matrix multiplication rules
    # query the network
    outputs = query(wih, who, inputs) # Test the network over the test set data (wih and who already set during training - no weight updating happens this time round)
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
        print("Accuracy ",sum(scorecard) / i,"\n")
    end
end

# Percentage Correct
percentage = sum(scorecard) / size(scorecard,1)

# Change log
# 4.22.2019 - added a % print function to check train / test accuracy
