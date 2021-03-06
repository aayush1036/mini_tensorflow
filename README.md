This project aims to ease the workflow of working with neural networks, I will be updating the code as I learn. 
I am currently pursuing bachelor's in data science and I am interested in Machine Learning and Statistics<br>
<a href='https://github.com/aayush1036/'> Github Profile </a><br>
<a href='https://aayushmaan1306.medium.com/'> Medium Profile</a><br>
<a href='https://aayush1036.github.io/profile_website/'>Website</a><br>
<a href='https://github.com/aayush1036/mini_tensorflow'>Github Repository of the source code</a><br>
<a href='https://pypi.org/project/MiniTensorflow/'>PyPI link of the project</a> <br>

This code contains two classes, one for the Layer and one for the Network <br>

# Layer Class
This class creates a Layer of the neural network which can be used for further calculations<br>
```python 
def __init__(self,inputs:np.array,n,activation = 'sigmoid',weights=None,bias=None,random_state=123,name=None) -> None:
```
Args:
1. name - The name of the layer, defaults to None
2. inputs - The inputs for the layer, shape = (n_x,m)
3. n - The number of neurons you would like to have in the layer
4. weights - The weights for the layer, initialized to random values if not given, shape = (n[l], n[l-1])
5. bias - The bias for the layer, initialized to random values if not given, shape = (n[l],1)
6. activation- The activation function you would like to use, defaults to sigmoid<br>
Can chose from ['sigmoid','tanh','relu']<br>
Raises ValueError if the activaion function is not among the specified functions<br>
Equations of activation functions for reference<br><img src="https://i.ibb.co/wpffTK6/Activation-Functions.png" alt="Activation-Functions" border="0">
7. random_state - The numpy seed you would like to use, defaults to 123
Returns: None<br>
Example <br>
```python
# goal - to create a layer of 5 neurons with relu activation function whose name is 'First hidden layer'and initializes random weights
x = np.random.randn(5,5)
layer1 = Layer(name='First hidden layer',inputs=x)
```
Fit function <br>
```python 
def fit(self)->np.array:
```
Fits the layer according to the activation function given to that layer<br>
For the process of fitting, it first calculates Z according to the equation <br>
<img src="https://i.ibb.co/q0YYBH5/eq1.png" alt="eq1" border="0"><br>
Then calculates the activation function by using the formula<br>
<img src="https://i.ibb.co/GCzDGKH/eq2.png" alt="eq2" border="0"><br>
Returns: np.array - Numpy array of the outputs of the activation function applied to the Z function
Example<br>

```python
# goal - you want to fit the layer to the activation function 
outputs = layer1.fit()
```

Derivative function <br>
```python 
def derivative(self)->np.array:
```
Calculates the derivative of the acivation function accordingly<br>
Returns: np.array - Numpy array containing the derivative of the activation function accordingly
Example <br>
```python
# goal - You want to calculate the derivative of the activation function of the layer
derivatives = layer1.derivative()
```
# Network Class
This class creates a neural network of the layers list passed to it<br>
```python
def __init__(self,layers:list) -> None:
```
Args:
1. layers - The list of layer objects 
Raises TypeError if any element in the layers list is not a Layer instance 
Example<br>
```python
# goal - To create a network with the following structure 
# Input layer - 2 neurons 
# First Hidden layer - 6 neurons with sigmoid activation function 
# Second Hidden Layer - 6 neurons with tanh activation function 
# Output Layer - 1 neuron with sigmoid activation function
X = np.random.randn(2,400)
layer1 = Layer('First hidden layer',n=6,inputs=X,activation='sigmoid')
layer2 = Layer('Second Hidden layer',n=6,activation='tanh',inputs=layer1.fit())
layer3 = Layer('Output layer',n=1,inputs=layer2.fit(),activation='sigmoid')
nn = Network([layer1,layer2,layer3])
```
Fit function <br>
```python
def fit(self)->np.array:
```
Propagates through the network and calcuates the output of the final layer i.e the output of the network <br>
Returns: np.array - The numpy  array containing the output of the network
Example<br>
```python 
# Goal- to propagate and find out the outputs of the network
outputs = nn.fit()
```

Compute cost function <br>
```python
def compute_cost(self,y:np.array,natural_log=True)->float:
```
Calculates the cost of the network compared to the target<br>

Args:<br>
1. y (np.array): Target values for the network 
2. natural_log (bool, optional): Whether you want to use log10 or natural log. Defaults to True.

Returns:<br>
float: The cost of that network

Example:<br>
``` python 
# Goal - to compute the cost for the network 
cost = nn.compute_cost(y=y)
```

Train function<br>
Trains the neural network for the specified iterations 
```python 
def train(self, epochs:int,history=False)->dict:
```
Args: <br>
1. epochs (int): The number of iterations for which you want to train the network 
2. history (bool, optional): If you want the history of gradients at each iterations. Defaults to False.

Returns: <br>
dict: The dictionary containing the gradients of parameters at each iteration 

Example:<br>

```python 
# Goal - to train the network for 500 epochs and get history at each epoch
nn.train(epochs=500, history=True)
```
Predict function <br>
Performs predictions on the given values 
```python 
def predict(self,values:np.array)->np.array:
```
Args:<br>
1. values (np.array): The values on which you want to predict 
Returns:<br>
np.array: The array of predictions

Attributes of a network<br>
1. params - The list containing total number of parameters initialized at each layer of the network
2. Summary<br>
Returns the summary of the network which is a pandas dataframe containing the following columns:<br>
* Layer Name: The name of the layer 
* Weights: The shape of the weights 
* Bias: The shape of the bias 
* Total Parameters: Total number of parameters initialized in the layer
Output<br>
<table>
<tr>
<td>Layer Name</td>
<td>Weights</td>
<td>Bias</td>
<td>Total parameters</td>
</tr>
<tr>
<td> First hidden layer</td>
<td> (6,2) </td>
<td> (6,1)</td>
<td> 18 </td>
</tr>
<tr>
<td> Second hidden layer </td>
<td> (6,6) </td>
<td> (6,1) </td>
<td> 42 </td>
</tr>
<tr>
<td> Output Layer </td>
<td> (1,6) </td>
<td> (1,1) </td>
<td> 7 </td>
</tr>
</table>