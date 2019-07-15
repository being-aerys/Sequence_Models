
# coding: utf-8

# In[2]:


import torch
import numpy as np
#print(torch.__version__)
torch.cuda.is_available()
t = torch.tensor([1,2,3])
dd = [[1,2,3],[4,5,6],[7,8,9]]
type(dd)
#print(dd)
#conver to a tensor from a list
dd_tensor = torch.tensor(dd)
type(dd_tensor)
#dd_tensor
x = dd_tensor.reshape(1,9)
#print(x)
y = x.reshape(3,3)
#print(y)
t_integer = torch.tensor([1,2,3])
t_float = torch.tensor([1.,2.,3.])
#t_sum = t_integer + t_float -------------cant do this, same dtype required for an operation
#print(t_sum)
a = (np.array([1,2,3,4,5,6,7,8,9]))
t1 = torch.Tensor(a) # constructor
t2 = torch.tensor(a)#factory funciton
t3 = torch.as_tensor(a)#factory funciton # accepts any list type, so if u wanna save memory use thisinstead of as_numpy
t4 = torch.from_numpy(a)#factory funciton #only accepts numpy arrays
print(t1)
print(t2)
print(t3)
print(t4)

#All three factory functions infer the data type from the input given
#the constructor converts into the global data type
a[0] = 0
#another difference is on how memory is allocated to these options of tensor creation
#first two create a new copy of the tensor while the later two just use a view of the numpy data
print(t1)#copy data
print(t2)#copy data
print(t3)#share data
print(t4)#share data

#so zero memory operation between numpy and tensor using the 3rd and the 4th method, maintain the same pointer forboth

#Tensor Operation types 
#reshaping operations, element-wise operations, reduction operations, access operations


# In[3]:


#squeeze function removes every dimension from the tensor whose length is 1
import torch
p1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
#p1.reshape(1,12) returns a reshaped value pf p1 but doesnt reshape p1 itself so make sure u override
#squeeze also return a squeezed version without changing the original tensor
p2 =p1.reshape(1,12)
print(p2)

#now count the number of square brackets after we squeeze
p3 = p2.squeeze()
print(p3)

#flatten function basically merges reshape and squeeze that we performed above

t = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(t.shape)
print(t.reshape(2,6))
def flatten_custom_func(tensor_arg):
    tensor_arg = tensor_arg.reshape(1,-1) # basically (1,x) where x k hunuparxa bhanera pytorch le aafai calculate garxa
    #depending on the first value (here 1) and the total number of elements
    tensor_arg = tensor_arg.squeeze()
    return tensor_arg
 
flatten_custom_func(p1)


# In[4]:



image_1 = torch.tensor([#1st dimension for the number of images in the batch, has 3 elements/ images as children
                        
                        [#1st image
                            # has 3 color channels as children
                         [#1st color: RED, has 4 pixel row values as children 
                         [1,1,1,1], #1st row, has 4 pixel column values as children
                         [1,1,1,1,],
                         [1,1,1,1],
                         [1,1,1,1]
                        ],
                         [#GREEN
                         [1,1,1,1],
                         [1,1,1,1,],
                         [1,1,1,1],
                         [1,1,1,1]
                        ],
                         [#BLUE
                         [1,1,1,1],
                         [1,1,1,1,],
                         [1,1,1,1],
                         [1,1,1,1]
                        ],
                        ],
    [#2nd image
        #has 3 colors as children
                         [#RED
                         [2,2,2,2],
                         [2,2,2,2,],
                         [2,2,2,2],
                         [2,2,2,2]
                        ],
                         [#GREEN
                         [2,2,2,2],
                         [2,2,2,2,],
                         [2,2,2,2],
                         [2,2,2,2]
                        ],
                         [#BLUE
                         [2,2,2,2],
                         [2,2,2,2,],
                         [2,2,2,2],
                         [2,2,2,2]
                        ],
                        ],
    [#3rd image
        #has 3 colors as children
                         [#RED
                         [3,3,3,3], 
                         [3,3,3,3,],
                         [3,3,3,3],
                         [3,3,3,3]
                        ],
                         [#GREEN
                         [3,3,3,3], 
                         [3,3,3,3,],
                         [3,3,3,3],
                         [3,3,3,3]
                        ],
                         [#BLUE
                         [3,3,3,3], 
                         [3,3,3,3,],
                         [3,3,3,3],
                         [3,3,3,3]
                        ],
                        ]
                       ])
print(image_1[0])#returns the first child of the root, i.e.,the first image as a whole
print(image_1[0][0])#returns the first child of the first image
print(image_1[0][0][0])#returns the first child of the first color
print(image_1[0][0][0][0])#returns the first child/ column of the first row


# In[5]:


image_1.reshape(1,-1)[0] #or t.flatten() or t.reshape(-1)---- we are flattening so that we can feed into a FC layer's node


# In[6]:


#However, we want to keep the outputs of the three images separately because we want individual
#predictions for each image
#do not flatten the different images together, because euta image as a whole send garn ho euta
#NN ko node ma instead of a single pixel from each image to one node

image_2 = image_1.flatten(start_dim = 1).shape# flatten for each element in the second axis
#flattened image constituting all the features of this image ( color channel, height, width of pixels)
print(image_2)


# In[7]:


b1 = torch.tensor([1,1])
b2 = torch.tensor([[1,1,],
                  [2,2]])
print(b1 + b2) #implicit broadcasting solved the problem
#however, what if b1 was of 1 * 3 dimesntions
b3 = torch.tensor([1,2,3])
# print(b3 + b2)cant do this because
#RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension


# In[8]:


#element wise = component wise = point wise operation
#reduction operations on a tensor are sum prod numel mean std
b2.sum()
#can reduce the tensor to a specific dimension too
to_reduce_tensor = torch.tensor([[1,2,3],
                               [4,5,6],
                               [7,8,9]])
to_reduce_tensor.sum (1) #you mess up everytime you work on dimensions
#you know the first axis i.e., axis/ dim = 0 is the row axis, tara ani tyo vertical line ho k axis
#row ta mathi bata tala janxa k re ni kta
#yesari samjhi, taking first axis means summing the elements of the first axis, damn sahi ho

print(image_1.max())

print(image_1.argmax())

