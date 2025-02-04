import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = x.shape[0]
  out = x.reshape(N, np.prod(x.shape[1:])).dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dw = x.reshape(x.shape[0], np.prod(x.shape[1:])).T.dot(dout)
  db = np.sum(dout,axis = 0)
  dx = dout.dot(w.T).reshape(x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  bool_x = x > 0
  dx = bool_x * dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    miu = np.mean(x, axis = 0)
    sigama = np.var(x, axis = 0)
    norm_x = (x - miu) / np.sqrt(sigama + eps)
    # print gamma.shape
    # print beta.shape
    out = norm_x * gamma + beta
    running_mean = momentum * running_mean + (1 - momentum) * miu
    running_var = momentum * running_var + (1 - momentum) * sigama
    cache = (momentum, miu, sigama, norm_x, gamma, beta, x, eps, N)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    norm_x = (x - running_mean) / np.sqrt(running_var + eps)
    out = norm_x * gamma + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  # cache = (momentum0, miu1, sigama2, norm_x3, gamma4, beta5, x6, eps7, N8)  #
  #############################################################################
  dgamma = np.sum(dout * cache[3], axis = 0)
  dbeta = np.sum(dout, axis = 0)
  dnorm_x = dout * cache[4]
  dvar = np.sum(dnorm_x * (cache[6] - cache[1]) * (-1 / 2.) * (cache[2] + cache[7])**(-3/2.), axis = 0)
  dmiu = np.sum((-1)*dnorm_x / np.sqrt(cache[2]+ cache[7]), axis = 0) + dvar * np.sum((-2) * (cache[6] - cache[1]),axis = 0)/cache[8]
  dx = dnorm_x / np.sqrt(cache[2]+ cache[7]) + dvar * 2*(cache[6] - cache[1]) / cache[8] + dmiu / cache[8]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  dgamma = np.sum(dout * cache[3], axis = 0)
  dbeta = np.sum(dout, axis = 0)
  dnorm_x = dout * cache[4]
  dvar = np.sum(dnorm_x * (cache[6] - cache[1]) * (-1 / 2.) * (cache[2] + cache[7])**(-3/2.), axis = 0)
  dmiu = np.sum((-1)*dnorm_x / np.sqrt(cache[2]+ cache[7])) + dvar * np.sum((-2) * (cache[6] - cache[1]),axis = 0)/cache[8]
  dx = dnorm_x / np.sqrt(cache[2]+ cache[7]) + dvar * 2*(cache[6] - cache[1]) / cache[8] + dmiu / cache[8]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p) / p
    out = mask * x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
  H_new = 1 + (H + 2 * pad - HH) / stride
  W_new = 1 + (W + 2 * pad - WW) / stride
  padded_x = np.pad(x, ((0,),(0,),(pad,),(pad,)), 'constant', constant_values = 0)
  out = np.zeros((N, F, H_new, W_new))
  for n in xrange(N):
    for f in xrange(F):
      for hn in xrange(H_new):
        for wn in xrange(W_new):
          # print padded_x[n, :, hn*stride:(hn*stride+HH), wn*stride:(wn*stride+WW)].shape
          # print w[f, :, :, :].shape
          out[n, f, hn, wn] = np.sum(padded_x[n, :, hn*stride:(hn*stride+HH), wn*stride:(wn*stride+WW)] * w[f, :, :, :]) + b[f] # a[1,:,3,4] != a[1][:][3][4]
      
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  N, F, H_new, W_new = dout.shape
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']
  H = w.shape[2]
  W = w.shape[3]
  padded_x = np.pad(x, ((0,),(0,),(pad,),(pad,)), 'constant', constant_values = 0)
  dx = np.zeros(padded_x.shape)
  dw = np.zeros(w.shape)
  # be careful about the implement for dx. It is enough for you to use your own knowlwdge to implement it.
  print 'x shape: ',x.shape
  print 'stride: ', stride
  for n in xrange(N):
    for f in xrange(F):
      for out_h in xrange(H_new):
        for out_w in xrange(W_new):
          # print dx[n, :, stride*out_h:(stride*out_h + H), stride*out_w:(stride*out_w + W)].shape
          # print w[f,:,:,:].shape
          dx[n, :, stride*out_h:(stride*out_h + H), stride*out_w:(stride*out_w + W)] += \
          w[f,:,:,:] * dout[n, f, out_h, out_w]
          #conbine to each other
          dw[f,:,:,:] += padded_x[n, :, stride*out_h:(stride*out_h + H), stride*out_w:(stride*out_w + W)]*\
          dout[n, f, out_h, out_w]
  dx = dx[:,:,pad:padded_x.shape[2]-pad,pad:padded_x.shape[3]-pad]
  # it is for dw
  
  # dw = np.zeros(w.shape)
  # for n in xrange(N):
  #   for f in xrange(F):
  #     for out_h in xrange(H_new):
  #       for out_w in xrange(W_new):
  #         dw[f,:,:,:] += padded_x[n, :, stride*out_h:(stride*out_h + H), stride*out_w:(stride*out_w + W)]*\
  #         dout[n, f, out_h, out_w]
  # it is for db
  db = np.sum(dout, axis = (0,2,3))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param, out_index)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  H_new = (H - pool_height) / stride + 1
  W_new = (W - pool_width) / stride + 1
  out = np.zeros((N, C, H_new, W_new))
  out_index = np.zeros(out.shape)
  for n in xrange(N):
    for c in xrange(C):
      for h_new in xrange(H_new):
        for w_new in xrange(W_new):
          a = x[n, c, h_new*stride:(h_new*stride + pool_height), w_new*stride:(w_new*stride + pool_width)]
          index = a.argmax()
          h_index = index / pool_width
          w_index = index % pool_height
          out_index[n, c, h_new, w_new] = index
          out[n, c, h_new, w_new] = a[h_index, w_index]


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param, out_index)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param, out_index) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param, out_index = cache
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  dx = np.zeros(x.shape)
  N, C, H_new, W_new = dout.shape
  for n in xrange(N):
    for c in xrange(C):
      for h_new in xrange(H_new):
        for w_new in xrange(W_new):
          index = out_index[n, c, h_new, w_new]
          h_index = index / pool_width
          w_index = index % pool_height
          dx[n, c, h_new*stride + h_index, w_new*stride + w_index] = dout[n, c, h_new, w_new]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  - running_mean = momentum * running_mean + (1 - momentum) * miu
    running_var = momentum * running_var + (1 - momentum) * sigama
    cache = (momentum, miu, sigama, norm_x, gamma, beta, x, eps, N)
  """

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, F, H, W = x.shape
  gamma_new = gamma.reshape(1,F,1,1)
  beta_new = beta.reshape(1,F,1,1) 
  running_mean = bn_param.get('running_mean', np.zeros((1,F,1,1), dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros((1,F,1,1), dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    miu = np.mean(x, axis = (0, 2, 3)).reshape(1,F,1,1)
    sigama = np.var(x, axis = (0, 2, 3)).reshape(1,F,1,1)
    norm_x = (x - miu) / np.sqrt(sigama + eps)
    # print gamma.shape
    # print beta.shape
    out = norm_x * gamma_new + beta_new
    running_mean = momentum * running_mean + (1 - momentum) * miu
    running_var = momentum * running_var + (1 - momentum) * sigama
    cache = (momentum, miu, sigama, norm_x, gamma, beta, x, eps, N)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    norm_x = (x - running_mean) / np.sqrt(running_var + eps)
    out = norm_x * gamma_new + beta_new
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None
  N, C, H, W = dout.shape
  gamma = cache[4].reshape(1,C,1,1)
  beta = cache[5].reshape(1,C,1,1)
  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                         cache[5]                          #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.        

  #cache = (momentum0, miu1, sigama2, norm_x3, gamma4, beta5, x6, eps7, N8)   #
  #############################################################################
  m = N*H*W
  dgamma = np.sum(dout * cache[3], axis = (0,2,3))
  dbeta = np.sum(dout, axis = (0,2,3))
  dnorm_x = dout * gamma
  dvar = np.sum(dnorm_x * (cache[6] - cache[1]) * (-1 / 2.) * (cache[2] + cache[7])**(-3/2.), axis = (0,2,3)).reshape(1,C,1,1)
  dmiu = np.sum((-1)*dnorm_x / np.sqrt(cache[2]+ cache[7]), axis = (0,2,3)).reshape(1,C,1,1) + dvar * np.sum((-2) * (cache[6] - cache[1]),axis = (0,2,3)).reshape(1,C,1,1)/m
  dx = dnorm_x / np.sqrt(cache[2]+ cache[7]) + dvar * 2*(cache[6] - cache[1]) / m + dmiu / m
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
