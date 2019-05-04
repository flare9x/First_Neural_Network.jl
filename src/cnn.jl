# Convolutional Neural Network
# Andrew Bannerman 5/2/2019

# RelU activation functions variants https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7 + Julia: https://int8.io/neural-networks-in-julia-hyperbolic-tangent-and-relu/
# Convolutional Neural Network: https://www.youtube.com/watch?v=YRhxdVk_sIs
# CNN - ground up https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
# CNN - http://cs231n.github.io/convolutional-networks/
# CNN - https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
# CNN - https://github.com/wiseodd/hipsternet/blob/master/hipsternet/neuralnet.py
# CNN layer julia = https://github.com/FluxML/Flux.jl/blob/master/src/layers/conv.jl

#----------------------
# max pooling expalined: https://www.youtube.com/watch?v=ZjM_XQa5s6s
# - Zero padding = https://www.youtube.com/watch?v=qSTv_m-KFk0
#----------------------

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
example = reshape(Float64.(scaled_train_inputs[5,2:size(scaled_train_inputs,2)]),784,1)
example = reshape(example,28,28)
heatmap(example,c = :greys)

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
m x n = image convolution block size
s = stride
p = padding
padding = "valid" = no padding, "same" = keep conv dim output same as input dim, "custom" = custom p value
Input should be divisible by 2
"""
m=3
n=3
s=1
p=3
function conv_layer(A::Array{Float64,2}; m::Int64=3,n::Int64=3, s::Int64=1, p::Int64=0, padding::String="valid") # mxn: block_size / p::String="valid"
# Note need to add a check so that the full conv box with stride fits into the image
# https://github.com/JuliaArrays/PaddedViews.jl/issues/9
  M,N = size(A)
  mc = M+(p*2)-m+1          # no. vertical blocks (rows) # adjusted for the padding dim
  nc = N+(p*2)-n+1          # no. horizontal blocks (cols)
  if padding == "same" # want output dim of conv to be same as input dims
  p=0 # reset p
  #Calculate the first conv layer output dimensions
  w = size(A,2)
  f = size(filt,2)
  #M_out,N_out = (M-f+1) , (N-f+1)
  conv_out_dim = Int64.((w-f+2*p) / s+1) # final convoltion output dim
  p = Int64.((f-1) / 2) # Calculate correct padding value to keep the output the same dims as the input
  pad = p+1 # this is first element position within pad in A for PaddedView() PaddedView.jl
    A_pad = Float64.(PaddedView(0.0, A,(M+(p*2),N+(p*2)), (pad,pad)))
elseif padding == "valid"
  p=0 # reset p
  pad = p+1 # this is first element position within pad in A
    A_pad = Float64.(PaddedView(0.0, A,(M+(p*2),N+(p*2)), (pad,pad)))
elseif padding == "custom"
    p = p # keep as function input
    pad = p+1 # this is first element position within pad in A
      A_pad = Float64.(PaddedView(0.0, A,(M+(p*2),N+(p*2)), (pad,pad)))
  end
  #A_pad = Float64.(PaddedView(0.0, A,(M+(p*2),N+(p*2)), (pad,pad))) # p*2 to increase dim in both directions pad,pad = position of the first element to pad around
  #B = Array{eltype(A)}(undef, m*n, mc*nc)
  M,N = size(A_pad)
  mc = M-m+1          # no. vertical blocks (rows)
  nc = N-n+1          # no. horizontal blocks (cols)
  B = zeros(Int64, m*n, 0)
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
    w = size(A_pad,2)
    f = size(filt,2)
    conv_out_dim = Int64.((w-f) / s+1) # This is conv output dimension (excluding padding)
    eltype(conv_out_dim)
    conv_out = zeros(conv_out_dim,conv_out_dim) # initialize output
    for j in 1:size(B,2) # loop over column dim (horz.)
      conv_out[j] = sum(filt .* reshape(B[:,j],m,n)) #+ bias # element wise multiplication
    end
         return conv_out
  end

  A = conv2D = conv_layer(A,s=1,p=3,padding="same")

  heatmap(conv_out,c = :greys)
m=2
n=2
s=2
pool_type = "max"
"""
    pooling_layer(A::Array{Float64,2}; m::Int64=2,n::Int64=2, s::Int64=1, pool_type::String="max")
m x n = image pooling block size
s = stride
pool_type = "max" | "avg"
"""
  function pooling_layer(A::Array{Float64,2}; m::Int64=2,n::Int64=2, s::Int64=1, pool_type::String="max")
    M,N = size(A)
    mc = M-m+1          # no. vertical blocks (rows)
    nc = N-n+1          # no. horizontal blocks (cols)
    B = zeros(Int8, m*n, 0) # output
    # Create custom index to allow for stride
    stride_col_index = collect(1:s:nc)
    stride_row_index = collect(1:s:mc)
    @inbounds for j = 1:size(stride_col_index,1) # Loop over columns horizontal
      @inbounds for i = 1:size(stride_row_index,1) # loop over rows vertical
          block = A[stride_row_index[i]:stride_row_index[i]+m-1, stride_col_index[j]:stride_col_index[j]+n-1]
          shapes = rotr90(reshape(block,1,m*n))
            B = hcat(B, shapes) # Append to output matrix
        end
    end
    h = size(A,1) # rows
    w = size(A,2) # cols
    f = m
    pooling_out_h = Int64.((h-f)/s+1) # rows
    pooling_out_w = Int64.((w-f)/s+1) # cols
    pooling_out = zeros(pooling_out_h,pooling_out_w) # initialize output
    if pool_type == "max"
    for j in 1:size(B,2) # loop over column dim (horz.)
      @inbounds pooling_out[j] = maximum(B[:,j]) #+ bias # element wise multiplication
    end
  elseif pool_type == "avg"
    for j in 1:size(B,2) # loop over column dim (horz.)
      @inbounds pooling_out[j] = mean(B[:,j]) #+ bias # element wise multiplication
    end
  end
    return pooling_out
  end

  pool_out = pooling_layer(A,m=2,n=2,s=2)
    heatmap(testing, c = :greys)

    conv2 = conv_layer(pool_out,s=1,p=3,padding="same")
heatmap(conv2, c = :greys)
  pool_out2 = pooling_layer(conv2,m=2,n=2,s=1)
heatmap(pool_out2, c = :greys)


"""
input ---> conv_layer ---> pooling ---> conv_layer ---> pooling ---> fully connected layer
"""
# Create Fully-Connected Layer
pooled_shape = size(pool_out2)[1] * size(pool_out2)[2]
flatten = reshape(pool_out2,pooled_shape,1) # flatten pool layer

# inpout for a normal NN below



       @time A = reshape(collect(1.0:1.0:81.0),9,:)
       @time B = DataFrame(conv(A,s=1,p=0))

       plots = conv(A,s=1,p=0)
       heatmap(plots)

       testing = DataFrame(A_pad)

       CSV.write("C:/Users/Andrew.Bannerman/Desktop/Julia/test_outB1.csv", testing;delim=',')
