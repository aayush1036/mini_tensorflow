import numpy as np
import pandas as pd 
np.random.seed(123)

class Layer:
    def __init__(self,name:str,inputs:np.array,n,activation = 'sigmoid',weights=None,bias=None) -> None:
        """Initializes the Layer with the given parameters

        Args:
            name (str): Name of the layer
            inputs (np.array): The inputs to the layer
            n ([type]): Number of neurons in the layer
            activation (str, optional): The activation function to use ['sigmoid','tanh']. Defaults to 'sigmoid'.
            weights ([type], optional): The weights for the neural network, choses random weights if not passed. Defaults to None.
            bias ([type], optional): The bias for the neural network, choses random bias if not passed. Defaults to None.
        """        
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.name = name
        if self.weights is None:
            self.weights = np.random.rand(n,inputs.shape[0])
        if self.bias is None:
            self.bias = np.random.rand(n,1)
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
        return a

class Network:
    def __init__(self,layers:list) -> None:
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
        print('Initialized the neural network')
    def fit(self)->np.array:
        """Propagates through the layers and returns the final output

        Returns:
            np.array: The array containing the output of the network
        """        
        for layer in self.layers[1:]:
            output = layer.fit()
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
        summary_df['Total parameters'] = total_params
        return summary_df
    @property
    def params(self):
        params_list = []
        for layer in self.layers:
            params_list.append(layer.weights.size + layer.bias.size)
        return params_list

if __name__ == '__main__':
    inputs = np.random.rand(64,10)
    layer1 = Layer(name='First hidden layer',inputs=inputs,n=4)
    layer2 = Layer(name='Second hidden layer',inputs=layer1.fit(),n=3)
    layer3 = Layer(name='Output layer',n=1,inputs=layer2.fit())
    nn_question = Network([layer1,layer2,layer3])
    print(nn_question.params)