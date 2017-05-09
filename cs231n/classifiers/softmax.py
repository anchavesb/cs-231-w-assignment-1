import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for train in range(num_train):
    scores = X[train].dot(W)
    scores -= np.max(scores) #Numerical stability

    sum_exp = np.sum(np.exp(scores))
    loss_train = np.exp(scores[y[train]])/ sum_exp
    loss += -np.log(loss_train)
    selector = [x for x in range(num_classes) if x != y[train]]
    #print("%s %s" % (y[train],loss_train))
    #print(selector)
    for aclass in range(num_classes):
      loss_class = np.exp(scores[aclass]) / sum_exp
      dW[:, aclass] += -((aclass == y[train]) - loss_class) * X[train]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]  # N
  scores = X.dot(W)  # W . X
  scores -= np.max(scores)

  sum_exp = np.sum(np.exp(scores), axis=1)
  loss_train = np.exp(scores[np.arange(num_train), y]) / sum_exp
  loss = np.sum(-np.log(loss_train))

  #print(scores[np.arange(num_train), y].shape)
  #print(scores.shape)
  #print(sum_exp[:,np.newaxis].shape)
  gradient = np.exp(scores) / sum_exp[:,np.newaxis]
  #print(gradient.shape)
  gradient[np.arange(num_train), y] = -(1 - gradient[np.arange(num_train), y])
  dW = X.T.dot(gradient)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

