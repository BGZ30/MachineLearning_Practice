import mnist
from convLayer import Conv3x3

training_set = mnist.training_images()
training_lables = mnist.training_lables()

conv = Conv3x3(8)  # conv layer with 8 filters
output = conv.forward(training_set[0])  # test for the first image

print(output.shape)