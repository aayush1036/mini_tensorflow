This project aims to ease the workflow of working with neural networks, I will be updating the code as I learn. 
I am currently pursuing bachelor's in data science and I am interested in Machine Learning and Statistics<br>
<a href='https://github.com/aayush1036/'> Github Profile </a><br>
<a href='https://aayushmaan1306.medium.com/'> Medium Profile</a><br>
<a href='https://aayush1036.github.io/profile_website/'>Website</a><br>

This code contains two classes, one for the Layer and one for the Network <br>

# Layer Class
This class creates a Layer of the neural network which can be used for further calculations
**__init__(self,inputs:np.array,n,activation = 'sigmoid',weights=None,bias=None,random_state=123,name=None) -> None**<br>
The constructor of the Layer class takes the following arguments:
1. name - The name of the layer, defaults to None
2. inputs - The inputs for the layer, shape = (n_x,m)
3. n - The number of neurons you would like to have in the layer
4. weights - The weights for the layer, initialized to random values if not given, shape = (n[l], n[l-1])
5. bias - The bias for the layer, initialized to random values if not given, shape = (n[l],1)
6. activation- The activation function you would like to use, defaults to sigmoid<br>
Can chose from ['sigmoid','tanh','relu']<br>
Raises ValueError if the activaion function is not among the specified functions<br>
Equations of activation functions for reference<br>
$ \\ \sigma(Z) = \frac{1}{1+e^{-Z}} \\ tanh(Z) = \frac{e^{Z} - e^{-Z}}{e^{Z} + e^{-Z}} \\ relu(Z) = max(0,Z)$<br> 
7. random_state - The numpy seed you would like to use, defaults to 123
Returns: None<br>
Example <br>
```python3
# goal - to create a layer of 5 neurons with relu activation function whose name is 'First hidden layer'and initializes random weights
x = np.random.randn(5,5)
layer1 = Layer(name='First hidden layer',inputs=x)
```

**fit(self)->np.array**<br>
Fits the layer according to the activation function given to that layer<br>
For the process of fitting, it first calculates Z according to the equation <br>
$Z^{[l]} = W^{[l]} \times X^{[l-1]} + b^{[l]}$<br>
Then calculates the activation function by using the formula<br>
$a^{[l]} = g^{[l]}(Z^{[l]})$
Returns: np.array - Numpy array of the outputs of the activation function applied to the Z function
Example<br>
```python3
# goal - you want to fit the layer to the activation function 
outputs = layer1.fit()
```

**derivative(self)->np.array**<br>
Calculates the derivative of the acivation function accordingly<br>
Returns: np.array - Numpy array containing the derivative of the activation function accordingly
Example <br>
```python3
# goal - You want to calculate the derivative of the activation function of the layer
derivatives = layer1.derivative()
```
# Network Class
This class creates a neural network of the layers list passed to it
**__init__(self,layers:list) -> None**<br>
The constructor of the Network class takes the following arguments:
1. layers - The list of layer objects 
Raises TypeError if any element in the layers list is not a Layer instance 
Example<br>
```python3
X = np.random.randn(2,400)
layer1 = Layer('First hidden layer',n=6,inputs=X)
layer2 = Layer('Second Hidden layer',n=6,activation='tanh',inputs=layer1.fit())
layer3 = Layer('Output layer',n=1,inputs=layer2.fit())
nn = Network([layer1,layer2,layer3])
```

**fit(self)->np.array**<br>
Propagates through the network and calcuates the output of the final layer i.e the output of the network <br>
Returns: np.array - The numpy  array containing the output of the network
Example<br>
```python3 
outputs = nn.fit()
```

**summary(self)->pd.DataFrame**<br>
Returns the summary of the network which is a pandas dataframe containing the following columns:<br>
1. Layer Name: The name of the layer 
2. Weights: The shape of the weights 
3. Bias: The shape of the bias 
4. Total Parameters: Total number of parameters initialized in the layer
Example<br>
```python3
summary = nn.summary()
print(summary)
```
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

Attributes of a network<br>
1. params - The list containing total number of parameters initialized at each layer of the network