# Convolutional Neural Network
# Andrew Bannerman 5/2/2019

# RelU activation functions variants https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7 + Julia: https://int8.io/neural-networks-in-julia-hyperbolic-tangent-and-relu/
# Convolutional Neural Network: https://www.youtube.com/watch?v=YRhxdVk_sIs
# CNN - ground up https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
# CNN - http://cs231n.github.io/convolutional-networks/
# CNN - https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
# CNN - https://github.com/wiseodd/hipsternet/blob/master/hipsternet/neuralnet.py
# CNN layer julia = https://github.com/FluxML/Flux.jl/blob/master/src/layers/conv.jl

# Needed packages
using DataFrames
using CSV
using Distributions
using Plots
using Random
using Images, CoordinateTransformations, TestImages, OffsetArrays, CoordinateTransformations # For data augmentation
using PaddedViews # for cropping images

# Train and test set data preparation

# Load MNIST Data set from .csv format
train_set = CSV.read("C:/Users/Andrew.Bannerman/Desktop/Julia/My_First_Neural_Network/Data/mnist_train.csv",header=false,types=fill(Float64,785))
test_set = CSV.read("C:/Users/Andrew.Bannerman/Desktop/Julia/My_First_Neural_Network/Data/mnist_test.csv",header=false,types=fill(Float64,785))

# Plot Example image
example = Float64.(train_set[2,2:size(train_set,2)])
example = rotl90(reshape(example,28,28))
heatmap(example)

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
scaled_train_inputs = DataFrame(hcat(train_label,m_train))
scaled_test_inputs = DataFrame(hcat(test_label,m_test))

# Plot the new scale example
example = reshape(Float64.(scaled_train_inputs[1,2:size(scaled_train_inputs,2)]),784,1)
example = reshape(example,28,28)
heatmap(example)

function train(wih::Array{Float64,2}, who::Array{Float64,2},inputs::Array{Float64,2}, targets::Array{Float64,2}; lr::Float64=0.1, drop_out::Bool=true, drop_out_p::Float64=.8) #wih, who, lr, inputs, targets)

  A = example

# Set weights and filter
# Initialize filter using a normal distribution with mean 0
# standard deviation inversely proportional the square root of the number of units
# Resource: https://stats.stackexchange.com/questions/200513/how-to-initialize-the-elements-of-the-filter-matrix
scale = 1.0
stdev = scale/sqrt(size(A,2)*(size(A,1)))
filt = reshape(rand(Normal(0, stdev), 9),3,3)

# Build the convolution layer
"""
    conv(A::Array{Float64,2}; m::Int64=3,n::Int64=3, s::Int64=1, p::Int64=1)
mxn = image convolution block size
s = stride
p = padding
"""
m=3
n=3
s=1
p=0
function conv(A::Array{Float64,2}; m::Int64=3,n::Int64=3, s::Int64=1, p::Int64=0) # mxn: block_size
# Note need to add a check so that the full conv box with stride fits into the image
# Conv = sum(filt .* conv) + bias
# https://github.com/JuliaArrays/PaddedViews.jl/issues/9
# Speed test
# Padding
  M,N = size(A)
  mc = M+(p*2)-m+1          # no. vertical blocks
  nc = N+(p*2)-n+1          # no. horizontal blocks
  padding = p+1
  A_pad = Float64.(PaddedView(0.0, A,(M+(p*2),N+(p*2)), (padding,padding))) # 1 layer of padding
  #B = Array{eltype(A)}(undef, m*n, mc*nc)
  B = zeros(Int8, m*n, 0)
  # Create custom index to allow for stride
  stride_col_index = collect(1:s:nc)
  stride_row_index = collect(1:s:mc)
  #B = Array{eltype(A)}(undef)
  # split the image into small convolution blocks
  # This function is similar to im2col
  i = 1
  j = 1
  @inbounds for j = 1:size(stride_col_index,1) # Loop over columns horizontal
    @inbounds for i = 1:size(stride_row_index,1) # loop over rows vertical
        block = A_pad[stride_row_index[i]:stride_row_index[i]+m-1, stride_col_index[j]:stride_col_index[j]+n-1]
        shapes = rotr90(reshape(block,1,m*n))
       B = hcat(B, shapes) # Append to output matrix
      end
  end
    # multiply the filter by the convolution block
    # Calculate spatial size of the volume
    # w = Input volume size
    # f = size of the Conv Layer neurons
    # s = Stride
    # p = Amount of Zero padding
    # Neuron fit = (W−F+2P)/S+1
#
    w = size(A,2)
    f = size(filt,2)
    conv_out_dim = Int64.((w-f+2*p) / s+1) # This is output dimension
    eltype(conv_out_dim)
    conv_out = zeros(conv_out_dim,conv_out_dim) # initialize output
    for j in 1:size(B,2) # loop over column dim (horz.)
      conv_out[j] = sum(filt .* reshape(B[:,j],m,n)) #+ bias # element wise multiplication
    end
         return conv_out
  end



       @time A = reshape(collect(1.0:1.0:25.0),5,:)
       @time B = DataFrame(conv(A,s=1,p=0))

       plots = conv(A,s=1,p=0)
       heatmap(plots)

       testing = DataFrame(conv_out)

       CSV.write("C:/Users/Andrew.Bannerman/Desktop/Julia/test_outB.csv", B;delim=',')
