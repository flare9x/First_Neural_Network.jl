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
example = Float64.(test_set[2,2:size(test_set,2)])
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
example = Float64.(scaled_train_inputs[1,2:size(scaled_train_inputs,2)])
example = rotl90(reshape(example,28,28))
heatmap(example)

# Build the convolution layer
function im2col(A, m,n) # mxn: block_size
    # Note need to add a check so that the full conv box with stride fits into the image
    # Conv = sum(filt .* conv) + bias
    # https://github.com/JuliaArrays/PaddedViews.jl/issues/9
    # Speed test
    # Padding
           M,N = size(A)
           mc = M-m+1          # no. vertical blocks
           nc = N-n+1          # no. horizontal blocks
           padding = p+1
          A = Int64.(PaddedView(0, A,(M+(p*2),N+(p*2)), (padding,padding))) # 1 layer of padding
          #B = Array{eltype(A)}(undef, m*n, mc*nc)
          B = zeros(Int8, m*n, 0)
        # Create custom index to allow for stride
              stride_col_index = collect(1:s:nc)
              stride_row_index = collect(1:s:mc)
           #B = Array{eltype(A)}(undef)
         @inbounds for j = 1:length(stride_col_index) # Loop over columns horizontal
             @inbounds for i = 1:length(stride_row_index) # loop over rows vertical
               block = A[stride_row_index[i]:stride_row_index[i]+m-1, stride_col_index[j]:stride_col_index[j]+n-1]
               shapes = rotr90(reshape(block,1,m*n))
               B = hcat(B, shapes) # Append to output matrix
              # @inbounds for k=1:m*n
                #  B[k,(j-1)*mc+i] = block[k] # Populate the output
                #if k == m*n
                #    i = i+1
                #    j = j+1
                #end
                print("j",j,"\ni",i,"\n")
               end
             end

             # Calculate spatial size of the volume
             # W = Input volume size
             # F = size of the Conv Layer neurons
             # S = Stride
             # P = Amount of Zero padding
             # Neuron fit = (W−F+2P)/S+1
                #conv_out = (W-F+2*P)/S+1 # This is output dimension conv * filt = 4*4
         #  end
           return B
       end

       @time A = reshape(1:81,9,:)
       @time B = DataFrame(im2col(A, 3,3))
