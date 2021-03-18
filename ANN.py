import numpy as np
import matplotlib.pyplot as plt

from functions import sigmoid, sigmoid_backward, relu, relu_backward
np.random.seed(1)

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary with parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
      
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
     
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


def linear_forward(A, W, b):
    """

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" or "linear"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
       
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".  
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    elif activation =="linear":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = Z
        activation_cache = None
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters, activation):
    """
    Default activation is relu for N-1 Layers and sigmoid for last layer. For full "linear" enter just enter 'linear'
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation[0])
        caches.append(cache)
       
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation[1])
    caches.append(cache)
   
    
    assert(AL.shape == (1,X.shape[1]))
           
    return AL, caches


def compute_cost(AL, Y):
    """
   

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = -(1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
   
    
    cost = np.squeeze(cost)      
    assert(cost.shape == ())
    
    return cost



def compute_MAE_loss(AL,Y):
    """ Compute MAE loss if we use a linear activation
    """
    m = Y.shape[1]
    
    loss = np.absolute(AL-Y)
    cost = np.divide(np.sum(loss), m)
    
    cost = np.squeeze(cost)
    
    assert(cost.shape==())
    
    return cost 


def compute_rien(AL, Y):
    
    """"
    Compute Huber loss if we use a linear activation
    
    
    """ 
    m = Y.shape[1]
    delta = 10
    cond = np.sum(np.absolute(Y-AL))
    if cond <= delta:
        cost = 0.5*np.sum((Y-AL)**2)
    else:
        cost = np.sum(delta* np.absolute(Y-AL)-0.5*delta**2)
    
    cost = np.squeeze(cost)
    assert(cost.shape ==())
    
    return cost

    
def compute_huber_loss(AL,Y):
    
    delta = 0.3
    loss = np.where(np.abs(Y-AL) < delta , 0.5*((Y-AL)**2), delta*np.abs(Y - AL) - 0.5*(delta**2))
    cost = np.sum(loss)
    cost = np.squeeze(cost)
    assert(cost.shape ==())
    return cost
    
    
def linear_backward(dZ, cache):
    """
   

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)* dZ.dot(A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims = True)
    dA_prev = W.T.dot(dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db



def linear_activation_backward(dA, cache, activation):
    """
   
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation[0] == "relu":
        
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
     
        
    elif activation[1] == "sigmoid":
        
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    elif activation[0] =='linear':
        dZ = dA
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
  
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, activation):
    """
    
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
 
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
 
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation)
  
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
    
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l+1)], current_cache, activation)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
       

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]- learning_rate * grads['db' + str(l+1)]
    #
    return parameters



def train_model(X_train, Y_train, X_test, Y_test, layers_dims, activation = ['relu', 'sigmoid'],
                loss = None, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, return_evaluation = True):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    if activation =='linear':
        activation = ['linear', 'linear']
        if loss:
            pass
        else:
            raise Exception('Missing one argument: loss. Choose a loss for linear activation,loss =  "huber_loss" or "MAE_loss" ')
       
    np.random.seed(1)
    train_costs = []                         # keep track of cost
    test_costs = []
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL_train, caches_train = L_model_forward(X_train, parameters, activation)
        AL_test, caches_test = L_model_forward(X_test, parameters, activation)
        
        
        # Compute binary crossentropy or MAE Loss 
        if activation[0] != 'linear':
            train_cost = compute_cost(AL_train, Y_train)
            test_cost = compute_cost(AL_test,Y_test)
        elif activation[0] == 'linear':
            if loss == 'MAE_loss':
                train_cost =compute_MAE_loss(AL_train, Y_train)
                test_cost =compute_MAE_loss(AL_test, Y_test)
            elif loss == 'huber_loss':
                train_cost =compute_huber_loss(AL_train, Y_train)
                test_cost =compute_huber_loss(AL_test, Y_test)
       
    
        # Backward propagation.
        grads = L_model_backward(AL_train, Y_train, caches_train, activation)
        
 
        # Update parameters.
       
        parameters =  update_parameters(parameters, grads, learning_rate)
       
                
        # Print the cost every 100 training example
    
        if print_cost and i % 100 == 0:
            print("train_cost after iteration %i: %f" %(i, train_cost))
            print("test_cost after iteration %i: %f" %(i, test_cost))
        
        if print_cost and i % 1 == 0:
            train_costs.append(train_cost)
            test_costs.append(test_cost)
    
    
    # plot the costs
    plt.figure(figsize = (16,9))
    plt.style.use('seaborn')
    plt.plot(np.squeeze(train_costs))
    plt.ylabel('cost')
    plt.plot(np.squeeze(test_costs))
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.legend(['train_cost', 'test_cost'])

    plt.show()
    
    # Print Evaluation
    
    eval_dict = evaluate(X_train, Y_train, X_test, Y_test, parameters, print_eval = True)
    
    return parameters, eval_dict



def predict(X, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    activation = ['linear','linear']
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    results = np.zeros((1,m))
    
    # Forward propagation
    results, caches = L_model_forward(X, parameters, activation)

    
    # convert probas to 0/1 predictions
    
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    
        
    return results

def evaluate(X_train, Y_train, X_test, Y_test, parameters, transpose = False, print_eval = True):
    if transpose == True:
        X_train = np.transpose(X_train)
        Y_train = np.transpose(Y_train)
        X_test = np.transpose(X_test)
        Y_test = np.transpose(Y_test)
    
    X = np.concatenate((X_train, X_test), axis = 1)
    Y = np.concatenate((Y_train, Y_test), axis = 1)
    
    results = predict(X, parameters)
    train_results = predict(X_train, parameters)
    test_results = predict(X_test, parameters )
    
    # Mean absolute error 
    MAE = np.sum(np.absolute(Y-results))/ Y.shape[1]
    train_MAE = np.sum(np.absolute(Y_train-train_results))/ Y_train.shape[1]
    test_MAE = np.sum(np.absolute(Y_test-test_results))/ Y_test.shape[1]
    
    # Mean Square Error 
    MSE = np.sum((Y-results)**2)/ Y.shape[1]
    train_MSE = np.sum((Y_train-train_results)**2)/ Y_train.shape[1]
    test_MSE = np.sum((Y_test-test_results)**2)/ Y_test.shape[1]
    
    #RMSE
    RMSE = np.sqrt(MSE)
    train_RMSE = np.sqrt(RMSE)
    test_RMSE = np.sqrt(RMSE)
    
    #Mean
    MEAN = np.mean(Y)
    
    
    #R2 
    R2 = 1- np.sum((Y-results)**2)/np.sum((Y-np.mean(Y))**2)
    
    
    eval_dict = {'MAE': MAE,
               'train_MAE': train_MAE,
               'test_MAE': test_MAE,
                'MSE': MSE,
                'train_MSE': train_MSE,
                'test_MSE': test_MSE,
                'RMSE': RMSE,
                'train_RMSE': train_RMSE,
                'test_RMSE': test_RMSE,
                'MEAN': MEAN,
                'R2': R2}
    
    for k, v in eval_dict.items():
        eval_dict[k] = round(v, 3)
    
    if print_eval:
        for k, v in eval_dict.items():
            print(f"{k:<15}{v:>15}")
            
    
    
    return eval_dict