# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# Importing the dataset and prepare the training set and the test set
#data set contains movies, users and ratings, ratings range from 1 star to 5 star

training_set = pd.read_csv('u1.base', delimiter = '\t')
test_set = pd.read_csv('u1.test', delimiter = '\t')
##convert the dataframe into array
training_set = np.array(training_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies

#in our data (training_set or test_set), the first column (column index 0) is the user ID, 
#and the second column (column index 1) is the movie ID, each user can rate multiple movies, 
#actually the third column (column index 2 ) is rating score of each user to different movies, 
# which ranges from 1 to 5 for the rating score 0, we say this movie is not watched by the user
# and it will not be counted in the process of training and testing stage

num_users = int(max(max(training_set[:,0]), max(test_set[:,0])))# the maximum number of the user ID
num_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))# the maximum number of the movie ID

# Converting the data into an array with users in lines and movies in columns
# because this data structure can be used to construct the torch tensors

# the folloing function converts the dataset (training_set or test_set) into the "list of list",
# which means an element in a list is still a list, e.g., training_set contains nb_users lists,
#and each list represents a user, which contains nb_movies ratings, an specific element in the 
#coverted array (matrix) repesent the rating score of a specific movie rated by a specific user.
#For this array (matrix) with users in lines and movies in columns, we can find there are many 0s 
#and very few positive number (rating) in each line (hence the matrix is sparse), which means only 
#small amount of movies are rated by users, 


def convert(data):
    new_data = []
    for id_users in range(1, num_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(num_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors of float point single type
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
nodes_first_hidden_layer=20
nodes_second_hidden_layer=10
nodes_third_hidden_layer=20
#nodes_fourth_hidden_layer=30
class SAE(nn.Module):## create a child class (SAE) which inherite from the parant class module (nn.Module)
    def __init__(self, ):#initial function
        super(SAE, self).__init__()#super function is able to use the methods and classes from the nn.module
#create an object which represents the full connection between the first input vector features and first encoded vector
        self.fc1 = nn.Linear(num_movies, nodes_first_hidden_layer)
#full connection between the first encoded vector and the first hidden layer       
        self.fc2 = nn.Linear(nodes_first_hidden_layer, nodes_second_hidden_layer)
#full connection between the first hidden layer and the second hidden layer 
        self.fc3 = nn.Linear(nodes_second_hidden_layer, nodes_third_hidden_layer)
#full connection between the second hidden layer and the output vector  
# the output vector has the same dimension as input vector, that is nb_movies nodes 
        self.fc4 = nn.Linear(nodes_third_hidden_layer, num_movies)

#        self.fc5 = nn.Linear(nodes_fourth_hidden_layer, nb_movies)
#sigmoid function is used to        
        self.activation = nn.Sigmoid()
# forward propagation
    def forward(self, x):
#Auto Encode the first full connection, obtain the first encoded vector
        x = self.activation(self.fc1(x))
#Auto Encode the second full connection, obtain the second encoded vector
        x = self.activation(self.fc2(x))
#Auto Encode the third full connection, obtain the third encoded vector
        x = self.activation(self.fc3(x))
#       x = self.activation(self.fc4(x))
#decoding the third full connection, no activation function required
        x = self.fc4(x)
        return x
sae = SAE()

#criterion for the loss function: mean square error
criterion = nn.MSELoss()

#optimizer is used to update the weight in each epoch
#actually, other optimizerz can also be applied in order to improve performance
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
#num_epoch is the number of epochs, the weights will be updated at each epoch
num_epoch =200
for epoch in range(1, num_epoch + 1):
    train_loss = 0
#s is the number of user who have already rate the movies.
#In order to save memory, do not do computations
# for the user who didn't rate any movies.
#Make s as float point to be consistent in te subsequent computation
    s = 0.
    for id_user in range(num_users):
#creat additional dimenson for the input for torch tensor computation
        input = Variable(training_set[id_user]).unsqueeze(0)
# keep the original input as target since the input is going to modify
        target = input.clone()

#Only looking at the users who rated at least one movie
#in order to save the memory. 
# target.data is all the movie ratings from this user
        if torch.sum(target.data > 0) > 0:
# stack auto encode the input ratings and obtain the output predicted ratings
            output = sae(input)
# Don't compute the gradient with respect to the target
# in order to save a lot of computations
            target.require_grad = False
    
#select the indexes of the original input which are 0
#and set the output of these corresponding indexes as 0
# in order to save the memory           
            output[target == 0] = 0
# compute the loss between the output and the original input (target)           
            loss = criterion(output, target)
 
# In target, since only the users who rated at least one movie are considered        
# mean_corrector is a coefficient used to make the computation of the loss function fairer          
# make sure the denominator is always larger than 0
            mean_corrector = num_movies/float(torch.sum(target.data > 0) + 1e-10)
            
# backward propagation is used to update the weights
# backward propagation decides the direction to which the weights
# will be updated, they will be increased or decreased            
            loss.backward()
# accumulate all the train loss for every user     
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            
#Apply the optimizer to update the weights  
#optimizer decides the intensity of the updates
# that is amount by which the weights will be updated 
            optimizer.step()
#(train_loss/s) is the average number of rating score deviation
# for instance, our original rating is 3 star, but the predicted rating is 2 star or 4 star
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(num_users):
# the reason to keep the training set instead of test set as the input is in the following
# nn model predicts the ratings of the movies that user has NOT watched
# we compare the predict ratings to the ratings of the test set because test contains the rating 
# of the movies that user has NOT watched  
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
 #  target = input.clone()
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = num_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item()*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))
