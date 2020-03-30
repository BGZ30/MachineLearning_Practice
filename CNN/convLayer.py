"""
    build a cnn layer
"""

import numpy as numpy

class Conv3x3:
    # a conv layer with 3x3 filters

    def __init__(self, numOfFilters):
        self.numOfFilters = numOfFilters

        # filters is a 3d array with dims [numOfFilters, 3, 3]
        # devide by 9 to reduce the variance
        self.filters = np.randn(numOfFilters, 3, 3) / 9

    
    # Convolution
    def get3x3Images(self, image):
        # (1): get the image dimensions
        h, w = image.shape

        # (2): get all possible 3x3 images
        for i in range(h-2):
            for j in range(w-2):
                img3x3 = image[i:(i+3), j:(j+3)]
                yield img3x3, i, j

    def forward(self, input):
        # input is the original 2d image
        h, w = input.shape

        # prepare the output 3d array, the image after applying the filters
        output = np.zeros((h-2, w-2, self.numOfFilters))

        # get all possible 3x3 images
        for img3x3, i, j in self.get3x3Images(input):
            output[i, j] = np.sum(img3x3 * self.filters, axis=(1, 2))

            """
            axis=(1, 2), which produces a 1d array of length numOfFilters where 
            each element contains the convolution result for the corresponding filter.
            """
            return output