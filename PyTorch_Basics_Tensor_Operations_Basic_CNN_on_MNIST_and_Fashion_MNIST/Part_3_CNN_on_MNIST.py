#Done to learn PyTorch basics from DeepLizard tutorials.
# coding: utf-8

# In[4]:


import torch 
import torchvision
import torchvision.transforms as transforms_package
torch.set_printoptions(linewidth = 120)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


#ETL : Extract from source, Transform into a proper tensor, Load as an object

#torchvision package gives us access to the links of different datasets that are available online
training_data = torchvision.datasets.MNIST(root = "../datasets/mnist",train = True, download = True, transform = transforms_package.Compose([transforms_package.ToTensor()]))

#load this extracted data into a DataLoader object
data_loader = torch.utils.data.DataLoader(training_data, batch_size = 10) #note that we have already defined the batch size here
#VVI data_loader itself contains all the data though
#now, each enumeration of data_loader contains 10 samples

#print(len(data_loader)) prints 6k not 60k since each data_loader enumeration has 10 samples

#to get the testing data, put the argument as train = False
testing_data = torchvision.datasets.MNIST(root = "../datasets/mnist_test",train = False, download = True, transform = transforms_package.Compose([transforms_package.ToTensor()]))


# In[6]:



epochs = 3 # can make it more than three but this model already converges in the second epoch
#can make data iteratble as batch = next(iter(data_loader)) so that the network can take a batch at a time
#print(batch)

#Steps for a NN implementaiton
#1.Extend the nn.module base class
#2.In the class constructor, Define layers as class attributes
#3.Use the network's layer attributes (as well as operations fron nn.functional api) to define the network's different layers' structures
#4.Implement the forward() method


#to build a NN, we import a torch.nn package
#Within in the nn package, there is a class called a module which is a base class for all nn modules
#all layers in pytorch extend the nn.Module class and inherit the attributes and functions
#this is called inhertitance
#with this inheritance, we get all the NN layers and the neural network as a whole
#why do we extend the layers and the whole network from a single class
#because the whole network can be assumed to be one large layer
#since the layers are essentially functions and the NN is a collection of functions ==> a funciton itself, this similarity is captured by the nn.Module class in the library
#thus we extend the nn.Module class if we want a new NN or a new layer withihn that NN 
#usually all neurons in a single layer are of the same type of neuron, so we define just 1 forward activation funciton and one weights tensor for the whole layer
#each layer has its own transformation and the composition of the forward pass
#thus every pytorch NN module/ Neural Noetwork has a forward method that represents the forward pass of the network as a whole
#when implementing the forward method, we use functions from nn.functional package that contains the activation, has an in-built a activation to which we just have to provide the input and the weights


# In[10]:



#each of the layer extends nn.modules class TORCH.NN.MODULES behind the scenes
#In reality, the actual definition given for a model in TORCH.NN.MODULES has the form==> class Linear(Module):

# class Module(object):
#        Base class for all neural network modules.

#        Your model as well as layers in your model should also subclass this class.

#     Modules can also contain other Modules, allowing to nest them in
#     a tree structure. You can assign the submodules as regular attributes::

#         import torch.nn as nn
#         import torch.nn.functional as F

#         class Model(nn.Module):
#             def __init__(self):
#                 super(Model, self).__init__()
#                 self.conv1 = nn.Conv2d(1, 20, 5)
#                 self.conv2 = nn.Conv2d(20, 20, 5)

#             def forward(self, x):
#                x = F.relu(self.conv1(x))
#                return F.relu(self.conv2(x))

#     Submodules assigned in this way will be registered, and will have their
#     parameters converted too when you call :meth:`to`, etc.


#you want to create a class that represents the NN as a whole and has all the layers as attributes 
#also, you want to implement a forward method that defines the forward propagation in your network


#Module bhanne class xa modules bhanne package bhitra Linear, Convolution named classes haru sangai 
#nn.Module has a forward function jun hamro layer le ra network as a whole le  implement garnu parxa
#But while we explicitly define a forward function for the network, we just provide certain required parameters needed for each layer instead of defining the forward() method ourself for each layer


import torch.nn as nn
class Network_dummy(nn.Module):
    def __init__(self):
        self.layer = None #single dummy layer inside the constructure
     
    def forward_propagation_demo(self, t): #dummy forward funciton takes in a tensor t ahnd tranforms it using the dummy layer
        
        return self.layer(t) #operation iong the input tenso t
    
    
        
class Network(nn.Module): 
    
    
    def __init__(self):
        super(Network,self).__init__() #initializes using the super class nn.Module, define layers as attributes
        
        #Linear and Conv2d classes extend the Module class thats why all the attrbutes that we have have a set of weights and a forward funciton of their own by default as inhetiance
        
        #parameters are subclasses that have a special property when used with Module class 
        #IMP : when assigned as class attribuites, they are automatically added as parameters of the module
        
      
        #kernel = filter, convolutional kernel = convolutional filter
        #out_channels = Number of filters to use in this layer
        #out_features (for linear layers) = Size of the output tensor for this linear layer, if it is the  last layer, it is kinda fixed already though        
        
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5) #when we say 5, it actually means a 5 * 5 kernel
        
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2) #2 * 2 
        
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)
        
        
        
        
        #How many hidden neurons to use? The no of the output neurons should be less than the no of input neurons, thus, the no of hideen neurns should be lss than the inpout nbeurons
        
        #This is how you flattten a batch of input to feed into a fully-connected layer in batch processing
        # You make a pair (batch_size, flattened tensor of each image)
        #Say if the image is 28*28 and has 3 channels from the previous layer then
        #(batch_size, 3 * 28*28) is the shape of the input to the fully connected layer, herr ramrosanga. shape(2,1 ) xa
        #there will be 3*28*28 neurons in this fully-connected layer
        #the no of output channels for this layer is actually the total no of neurons that will be present in the next layer
        
        
        
        #where did the  8 * 8 come from?#that is the length of the flattened tensor from the previous layer

        self.fully_connected1 = nn.Linear(in_features = 12 * 8 * 8, out_features = 120) #see we dont even have to think about the inconvenience of the batch size
        self.fully_connected2 = nn.Linear(in_features = 120, out_features = 60)
        self.output_layer = nn.Linear(in_features = 60, out_features = 10)
        
    
        
        #usually, with a conv layer, we increase the size of the out channels
        #with a linear layer, we decrease
        #each of the layers should have a set of weights and the definition for the forward function
        #Since we are extending Module class, each of them has a set of weight and a forward function already defined in the definiotion of Module
        #nn Module keeps track of the weight tensors in each layer  and by extending the Module , we inherit the functionality automatially
        
    def flat_features_except_batch(self, x):
        size = x.size()[1:]  #get the shape except for the first axis i.e., no of examples in the batch to flatten all dimensions except the batch dimension
        num_features_per_example_as_seen_by_the_fully_connected_layer = 1
        for s in size:       # Get the products
            num_features_per_example_as_seen_by_the_fully_connected_layer *= s
        return num_features_per_example_as_seen_by_the_fully_connected_layer
    
    def forward(self, t): #dummy forward funciton takes in a tensor t ahnd tranforms it using the dummy layer
        out = self.conv1(t)
        #print(out[0])
        out = self.maxpool1(out)
        out = self.conv2(out)
        # print(out.data.shape) returns torch.Size([10, 12, 8, 8]), so batch size handled automatically
#         import time
#         time.sleep(1000)
        out = out.view(-1, self.flat_features_except_batch(out)) #The view function is meant to reshape the tensor.
        #-1 means next parameter jati a esko anusaar yo dimension milai
        #you will have a 16 depth feature map. You have to flatten this to give it to the fully connected layer. So you tell pytorch to reshape the tensor you obtained to have specific number of columns and tell it to decide the number of rows by itself.
        
        
#         What is the meaning of parameter -1?
#         If there is any situation that you don't know how many rows you want but are sure of the number of columns, then you can specify this with a -1. (Note that you can extend this to tensors with more dimensions. Only one of the axis value can be -1). This is a way of telling the library: "give me a tensor that has these many columns and you compute the appropriate number of rows that is necessary to make this happen".

#         This can be seen in the neural network code that you have given above. After the line x = self.pool(F.relu(self.conv2(x))) in the forward function, you will have a 16 depth feature map. You have to flatten this to give it to the fully connected layer. So you tell pytorch to reshape the tensor you obtained to have specific number of columns and tell it to decide the number of rows by itself.

#         Drawing a similarity between numpy and pytorch, view is similar to numpy's reshape function.
        
        
        
        #print(out.shape)
        out = self.fully_connected1(out)
        out = self.fully_connected2(out)
        #print(out.size)
        out = self.output_layer(out)
        #print("final output ko shape yesto ")
        #print(out.size)
        return out #final output of the network
  


# In[11]:



#you can use padding as an argument to preserve the original size of the height and the width of the data after it decreases because of the convolution
#get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

network = Network()# 
print(network)
# print(network.conv1)
# print(network.conv1.weight)
#weight is an instance of Parameter class that extends tensor.Tensor class and represents the learnable parameters of each layer
#print(network.conv1.weight.shape) #this returns the tensor that represents all the filters of this conv layer
#we can access a single filter as network.conv2.weight[0]
#weight of a convolution filter is the filter itself 
#define the loss function you want to use for your network, provided in torch.optim package
loss_fn = nn.CrossEntropyLoss()

#optimize using some optimization method, here SGD/ Stochastic Gradient Descent
optimizer = torch.optim.SGD(network.parameters(), lr = 0.01)


# In[13]:


#actual operation of CNN

count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(epochs):
    print("Epoch is ",epoch)
    #print("data loader size is : ",len(data_loader)) #VVI : data_loader lai enumerate garyo bhane harek enumeration ma 10 ota examples hunxa
    for i, (features, labels) in enumerate (data_loader): #enumerate indexes a list of tuples to make looping easier
        
        #print("training count is ",i)
        #print("\n\n")
        total_correct_predictions = 0
        
        
    #     print("Input shape is : ")
    #     print(features.shape)
    #     print("labels shape is :")
    #     print (labels.shape)
    #     print(labels.data)
    
    
        # Clear gradients
        optimizer.zero_grad()
        #print("features shape ",features.shape)
        
        # Forward propagation
        train_pass = features.view(10,1,28,28)
 
        final_output = network(train_pass) #network operation
    
        # Calculate softmax and ross entropy loss
        loss = loss_fn(final_output, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        count += 1
        
        
        ###################EVERY 100 iterations###########################
        if (count +1)%100 ==0:
            correct = 0
            incorrect = 0
            total=0
            test_count = 0
            for (test_features, labels) in testing_data:
                test_count = test_count+1
                #print("test count ",test_count)
                #print("test_features size ",len(testing_data))
                test = test_features.view(1,1,28,28)
                output = network(test)
                _,predicted = torch.max(output.data,1)
                #print("here " ,labels.data)
                #print("predicted and labels.data")
#                 print("predicted is", predicted.data)
#                 print("label is ",labels.data)
#                 print(" equality check ",(predicted == labels.data).sum())
                correct += (predicted == labels).sum()
                #print("correct is ",correct.data)
           
            accuracy = 100 *correct/float(10000)
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            
            print('Iteration Number : {}, Training Loss: {}, Testing Accuracy: {}%'.format(count,loss.data,accuracy))
        
        ######################################################################
          
    


# In[ ]:


# visualization of the training loss 

plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iterations")
plt.ylabel("Training Loss")
plt.title("Loss vs Number of iterations")
plt.show()

# visualization of the accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iterations")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of iteration")
plt.show()

