import numpy as np
import pandas as pd
class Layer:
    def __init__(self,inputs:np.array,n,activation = 'sigmoid',weights=None,bias=None,random_state=123,name=None) -> None:
        """Initializes the Layer with the given parameters

        Args:
            name (str): Name of the layer, defaults to None
            inputs (np.array): The inputs to the layer
            n ([type]): Number of neurons in the layer
            activation (str, optional): The activation function to use ['sigmoid','tanh']. Defaults to 'sigmoid'.
            weights ([type], optional): The weights for the neural network, choses random weights if not passed. Defaults to None.
            bias ([type], optional): The bias for the neural network, choses random bias if not passed. Defaults to None.
        """    
        np.random.seed(random_state)    
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.name = name
        self.random_state = random_state
        if self.weights is None:
            self.weights = np.random.randn(n,inputs.shape[0])
        if self.bias is None:
            self.bias = np.zeros((n,1))
        if activation.strip().lower() not in ['sigmoid','tanh','relu','softmax']:
            raise ValueError('Activation should be among [sigmoid,tanh,relu,softmax]')
        else:
            self.activation = activation
    def __sigmoid(self,x:np.array)->np.array:
        """Sigmoid activation function for the neural network
        Calculates sigmoid value by using the formula sigmoid(z) = 1/(1+e^(-z))

        Args:
            x (np.array): The array of values on which you want to apply sigmoid 

        Returns:
            np.array: The array of sigmoid values
        """        
        return 1/(1+np.exp(-x))
    def __relu(self,x:np.array)->np.array:
        """Returns the ReLU function applied on that array

        Args:
            x (np.array): The array on which you want to apply the ReLU function on

        Returns:
            np.array: The array with ReLU function applied 
        """              
        return np.maximum(0,x)
    def derivative(self)->np.array:
        """Returns the differentiation of the activation function for a corresponding layer

        Returns:
            np.array: The array containing the derivative of the activation function
        """        
        a = self.fit()
        if self.activation.strip().lower() == 'sigmoid':
            return a*(1-a)
        elif self.activation.strip().lower() == 'tanh':
            return 1-(a**2)
        elif self.activation.strip().lower() == 'relu':
            return np.int64(a>0)

    def fit(self)->np.array:
        """Fits the layer according to the formula a = activation_function(wx+b)

        Returns:
            np.array: The output of the activation function for that layer
        """        
        z = np.dot(self.weights, self.inputs) + self.bias
        if self.activation.strip().lower() == 'sigmoid':
            a = self.__sigmoid(z)
        elif self.activation.strip().lower() == 'tanh':
            a = np.tanh(z)
        elif self.activation.strip().lower() == 'relu':
            a = self.__relu(z)
        elif self.activation.strip().lower() == 'softmax':
            a = np.exp(z)/np.exp(z).sum(axis=0)
        return a
class Network:
    def __init__(self,layers:list,y:list) -> None:
        """Initializes the neural network with the given layers

        Args:
            layers (list): List of layers in the network

        Raises:
            TypeError: Raises a TypeError if any of the layers in the layers list is not a Layer instance
        """            
        self.layers = layers
        for layer in layers:
            if not isinstance(layer,Layer):
                raise TypeError('All the values in the layers list should by Layer instances')
        self.y = y
    def fit(self,get_history=False)->tuple:
        """Propagates through the network and returns the output and history of outputs if specified

        Args:
            get_history (bool, optional): Whether you want outputs of all layers or not. Defaults to False

        Returns:
            tuple: A tuple containing the output of the network and history if specified
        """                     
        memory = {}
        for layer in self.layers:
            output = layer.fit()
            memory[layer.name] = output
        if get_history:
            return output, memory
        else:
            return output
    def summary(self)->pd.DataFrame:
        """Returns the DataFrame containing the summary of the network passed to it

        Returns:
            pd.DataFrame: The DataFrame containing the summary of the network
        """        
        summary_df = pd.DataFrame()
        layer_name = []
        layer_weights = []
        layer_bias = []
        total_params = [] 
        for layer in self.layers:
            layer_name.append(layer.name)
            layer_weights.append(layer.weights.shape)
            layer_bias.append(layer.bias.shape)
            total_params.append(layer.weights.size+layer.bias.size)
        summary_df['Layer Name'] = layer_name
        summary_df['Weights'] = layer_weights
        summary_df['Bias'] = layer_bias
        summary_df['Total Parameters'] = total_params
        return summary_df
    @property
    def params(self)->list:
        """Gets the total number of parameters in each layer

        Returns:
            params_list: The list containing the total number of parameters in each layer
        """        
        params_list = []
        for layer in self.layers:
            params_list.append(layer.weights.size+layer.bias.size)
        return params_list
    def compute_cost(self,y:np.array,natural_log=True)->float:
        """Calculates the cost of the network compared to the target

        Args:
            y (np.array): Target values for the network 
            natural_log (bool, optional): Whether you want to use log10 or natural log. Defaults to True.

        Returns:
            float: The cost of that network
        """               
        outputs = self.fit()
        if natural_log:
            cost = -np.mean((y*np.log(outputs))+((1-y)*np.log(1-outputs)))
        else:
            cost = -np.mean((y*np.log10(outputs))+((1-y)*np.log10(1-outputs)))
        return cost
    def backward_propaagation(self):
        output = self.fit()
        d_cost = -self.y/output + (1-self.y)/(1-output)
        prod = d_cost*self.layers[-1].fit()*self.layers[-1].weight
        layers_copy = self.layers[:-1].copy()