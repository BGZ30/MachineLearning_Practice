"""
    Neighboring pixels in images tend to have similar values, so conv layers 
    will typically also produce similar values for neighboring pixels in 
    outputs. As a result, much of the information contained in a conv layer’s
    output is redundant.

    Pooling layers solve this problem. All they do is reduce the size of the input it’s given
    by pooling values together in the input. The pooling is usually done 
    by a simple operation like max, min, or average.

    Pooling divides the input’s width and height by the pool size.
"""

import numpa as np

"""
    For MNIST CNN, place a Max Pooling layer with a pool size of 2 right 
    after the the initial conv layer.
    The pooling layer will transform a 26x26x8 input into a 13x13x8 output.
"""

class MaxPool2:
    # get all 2x2 images 'non-overlapping' to pool over
    def getImages2x2(self, image):
        h, w = image.shape
        new_h = h//2
        new_w = w//2

        for i in range(new_h):
            for j in range(new_w):
                images2x2 = image[(2*i):(2*i+2), (2*j):(2*j+2)]  # non-overlapping
                yield images2x2, i, j
    
    # perform the pooling 
    # return a 3d array with dimensions (h / 2, w / 2, numOfFilters).
    # input is a 3d array with dimensions (h, w, numOfFilters), from the conv layer
    def forward(self, input):
        h, w, numOfFilters = input.shape
        output = np.zeros(h//2, w//2, numOfFilters)

        for img2x2, i, j in self.getImages2x2(input):
            output[i, j] = np.amax(img2x2, axiz=(0, 1))

    return output
