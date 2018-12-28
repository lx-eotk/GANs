'''
Generator
input = (100, )
Dense(128*16*16, ‘relu’)
Reshape((16, 16, 128))
Upsampling
Conv2D(128, kernel = 4)
Relu
Upsampling
Conv2D(64, kernel = 4)
Relu
Conv2D(3, kernel = 4)
tanh

Training
Adam(lr = 0.0002, beta = 0.5)
'''