import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    def compute_size(H,W,filter_size, stride,pool_size,pool = True):
        pad = (filter_size - 1) / 2 # this set is for H_out == H
        H_out = (H - filter_size + 2*pad) / stride + 1
        W_out = (W - filter_size + 2*pad) / stride + 1
        if pool:
            H_out /= pool_size
            W_out /= pool_size
        return (H_out, W_out)

    stride = 1

    self.params['W1'] = np.random.randn(num_filters,input_dim[0],filter_size,filter_size)*weight_scale
    self.params['b1'] = np.zeros(num_filters)

    H_out, W_out = compute_size(input_dim[1],input_dim[2], filter_size, stride, 2)
    self.params['W2'] = np.random.randn(num_filters*H_out*W_out, hidden_dim) * weight_scale
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b3'] = np.zeros(num_classes)
                                                    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #conv
    conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    #conv to affine
    N, C, H_out, W_out = conv_out.shape 
    out_new = conv_out.reshape(N, C*H_out*W_out)
    # print conv_out.shape
    # print W2.shape
    hidden_a, hidden_cache = affine_relu_forward(out_new, W2, b2)
    #hidden to classes_score
    scores, scores_cache = affine_forward(hidden_a, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    loss += self.reg*0.5 * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2)+np.sum(self.params['W3']**2))

    dhidden_a, dW3, db3 = affine_backward(dout, scores_cache)
    dout_new, dW2, db2 =affine_relu_backward(dhidden_a, hidden_cache)
    dconv_out = dout_new.reshape(N, C, H_out, W_out)
    dx, dW1, db1 = conv_relu_pool_backward(dconv_out, conv_cache)
    grads['W1'] = dW1 + self.reg * self.params['W1']
    grads['W2'] = dW2 + self.reg * self.params['W2']
    grads['W3'] = dW3 + self.reg * self.params['W3']
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
class MultiplyConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
  [conv-relu-pool]XN - [affine]XM - [softmax or SVM]
  [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=5,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm = True ):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    def compute_size(H,W,filter_size, stride,pool_size,pool = True):
        pad = (filter_size - 1) / 2 # this set is for H_out == H
        H_out = (H - filter_size + 2*pad) / stride + 1
        W_out = (W - filter_size + 2*pad) / stride + 1
        if pool:
            H_out /= pool_size
            W_out /= pool_size
        return (H_out, W_out)

    #default:
    #stride = 1
    #pool_size = 2
    #pool_stride = 2


    F = num_filters
    C = input_dim[0]
    #for x
    H = input_dim[1]
    W = input_dim[2]

    self.params['W1'] = np.random.randn(F,C,filter_size,filter_size)*weight_scale
    self.params['b1'] = np.zeros(F)

    self.params['W2'] = np.random.randn(F,F,filter_size,filter_size)*weight_scale
    self.params['b2'] = np.zeros(F)

    self.params['W3'] = np.random.randn(F,F,filter_size,filter_size)*weight_scale
    self.params['b3'] = np.zeros(F)

    self.params['W4'] = np.random.randn(F,F,filter_size,filter_size)*weight_scale
    self.params['b4'] = np.zeros(F)

    out_H = H / 4
    out_W = W / 4
        
    self.params['W5'] = np.random.randn(F*out_W*out_H,hidden_dim)*weight_scale
    self.params['b5'] = np.zeros(hidden_dim)

    self.params['W6'] = np.random.randn(hidden_dim,num_classes)*weight_scale
    self.params['b6'] = np.zeros(num_classes) 

    self.bn_params = []
    if self.use_batchnorm:
        self.bn_params = [{'mode': 'train'} for i in xrange(6)]
        for i in xrange(4):
            gamma_ = 'gamma' + str(i+1)
            beta_ = 'beta' + str(i+1)
            gamma = {gamma_:np.ones(F)}
            beta = {beta_:np.zeros(F)}
            self.params.update(gamma)
            self.params.update(beta)
        gamma = {'gamma5':np.ones(hidden_dim)}
        beta = {'beta5':np.zeros(hidden_dim)}
        self.params.update(gamma)
        self.params.update(beta)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W, b = [0 for i in xrange(10)],[0 for i in xrange(10)]
    gamma, beta = [0 for i in xrange(10)],[0 for i in xrange(10)]
    for i in xrange(1,7,1):
        W_ = 'W'+str(i)
        b_ = 'b'+str(i)
        W[i], b[i] = self.params[W_], self.params[b_]
    for i in xrange(1,6,1):
        gamma_ = 'gamma'+str(i)
        beta_ = 'beta'+ str(i)
        gamma[i], beta[i] = self.params[gamma_], self.params[beta_]
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W[1].shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #conv_relu_BN_conv_relu_pool1
    conv_out1, conv_cache1 = conv_relu_forward(X, W[1], b[1], conv_param)
    conv_relu_BN1, conv_relu_BN_cache1 = spatial_batchnorm_forward(conv_out1, gamma[1], beta[1], self.bn_params[1])
    conv_out2, conv_cache2 = conv_relu_pool_forward(conv_relu_BN1, W[2], b[2], conv_param, pool_param)

    #BN_conv_relu_conv_relu_pool1
    conv_relu_BN2, conv_relu_BN_cache2 = spatial_batchnorm_forward(conv_out2, gamma[2], beta[2], self.bn_params[2])
    conv_out3, conv_cache3 = conv_relu_forward(conv_relu_BN2, W[3], b[3], conv_param)
    conv_relu_BN3, conv_relu_BN_cache3 = spatial_batchnorm_forward(conv_out3, gamma[3], beta[3], self.bn_params[3])
    conv_out4, conv_cache4 = conv_relu_pool_forward(conv_relu_BN3, W[4], b[4], conv_param, pool_param)

    #BN_reshape_affine_relu_BN_affine_softmax
    conv_relu_BN4, conv_relu_BN_cache4 = spatial_batchnorm_forward(conv_out4, gamma[4], beta[4], self.bn_params[4])
    N, C, H_out, W_out = conv_relu_BN4.shape 
    conv_relu_BN4_new = conv_relu_BN4.reshape(N, C*H_out*W_out)

    hidden_a, hidden_cache = affine_relu_forward(conv_relu_BN4_new, W[5], b[5])
    affine_relu_BN5, affine_relu_BN_cache5 =  batchnorm_forward(hidden_a, gamma[5], beta[5], self.bn_params[5])
    #hidden to classes_score
    scores, scores_cache = affine_forward(affine_relu_BN5, W[6], b[6])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    for i in xrange(1,7,1):
        loss+=self.reg*0.5*np.sum(W[i]**2)
    #round three
    daffine_relu_BN5, dW6, db6 = affine_backward(dout, scores_cache)
    dhidden_a, dgamma5, dbeta5 = batchnorm_backward(daffine_relu_BN5, affine_relu_BN_cache5)

    dconv_relu_BN4_new, dW5, db5 =affine_relu_backward(dhidden_a, hidden_cache)
    dconv_relu_BN4 = dconv_relu_BN4_new.reshape(N, C, H_out, W_out)
    dconv_out4, dgamma4, dbeta4 = spatial_batchnorm_backward(dconv_relu_BN4, conv_relu_BN_cache4)

    #round two
    dconv_relu_BN3, dW4, db4 = conv_relu_pool_backward(dconv_out4, conv_cache4)
    dconv_out3, dgamma3, dbeta3 = spatial_batchnorm_backward(dconv_relu_BN3, conv_relu_BN_cache3)
    dconv_relu_BN2, dW3, db3 = conv_relu_backward(dconv_out3, conv_cache3)

    #round one
    dconv_out2, dgamma2, dbeta2 = spatial_batchnorm_backward(dconv_relu_BN2, conv_relu_BN_cache2)
    dconv_relu_BN1, dW2, db2 = conv_relu_pool_backward(dconv_out2, conv_cache2)
    dconv_out1, dgamma1, dbeta1 = spatial_batchnorm_backward(dconv_relu_BN1, conv_relu_BN_cache1)
    dx, dW1, db1 = conv_relu_backward(dconv_out1, conv_cache1)

    grads['W1'] = dW1 + self.reg * self.params['W1']
    grads['W2'] = dW2 + self.reg * self.params['W2']
    grads['W3'] = dW3 + self.reg * self.params['W3']
    grads['W4'] = dW4 + self.reg * self.params['W4']
    grads['W5'] = dW5 + self.reg * self.params['W5']
    grads['W6'] = dW6 + self.reg * self.params['W6']
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    grads['b4'] = db4
    grads['b5'] = db5
    grads['b6'] = db6

    grads['gamma1'] = dgamma1
    grads['gamma2'] = dgamma2
    grads['gamma3'] = dgamma3
    grads['gamma4'] = dgamma4
    grads['gamma5'] = dgamma5
    grads['beta1'] = dbeta1
    grads['beta2'] = dbeta2
    grads['beta3'] = dbeta3
    grads['beta4'] = dbeta4
    grads['beta5'] = dbeta5

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

