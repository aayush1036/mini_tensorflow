from MiniTensorflow import Network, Layer
import numpy as np
X = np.random.randn(2,3)
y = (np.random.randn(1,3)>0)

def layer_size(X,Y):
  n_x=X.shape[0]
  n_hidden = int(input('Enter the number of hidden layers you want '))
  index_list = []
  number_list = []
  for i in range(n_hidden):
    index_list.append(i+1)
    number_list.append(int(input(f'Enter the number of neurons in hidden layer {i+1} ')))
  n_h = dict(zip(index_list, number_list))
  n_y=Y.shape[0]
  return(n_x,n_h,n_y)

input_layer, hidden_layer, output_layer = layer_size(X,y)

def int_value(n_x,n_h,n_y):
  np.random.seed(1)
  W1 = np.random.randn(n_h[1], n_x)*0.01
  b1 = np.zeros((n_h[1],1))
  W2 = np.random.randn(n_h[2],n_h[1])*0.01
  b2 = np.zeros((n_h[2],1))
  W3 = np.random.randn(n_h[3], n_h[2])*0.01
  b3 = np.zeros((n_h[3],1))
  W4 = np.random.randn(n_y, n_h[3])*0.01
  b4 = np.zeros((n_y,1))
  parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3, "b3":b3,"W4":W4,"b4":b4}
  return parameters

params = int_value(input_layer, hidden_layer, output_layer)

layer1 = Layer(name='First Hidden Layer',n=6,weights=params['W1'],bias=params['b1'],inputs=X,activation='relu',random_state=1)
layer2 = Layer(name='Second Hidden Layer',n=4,weights=params['W2'],bias=params['b2'],inputs=layer1.fit(),activation='relu',random_state=1)
layer3 = Layer(name='Third Hidden Layer',n=2,weights=params['W3'],bias=params['b3'],inputs=layer2.fit(),activation='tanh',random_state=1)
layer4 = Layer(name='Output Layer',n=1,weights=params['W4'],bias=params['b4'],inputs=layer3.fit(),activation='sigmoid',random_state=1)

network = Network([layer1,layer2,layer3,layer4])
print(network.compute_cost(y=y,natural_log=True))