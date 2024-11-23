import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

###################### Designing an ANN architectures #########################

### MLP architecture
class MLP(nn.Module): # All models should inherit from nn.Module
    # This part can be changed based on the design decision.
    def __init__(self, num_input, hidden1_size, num_classes): # Define our ANN structures here
        super(MLP, self).__init__()
        # nn.Linear(in_features, out_features, bias): y = w^Tx + bias
        self.hidden1 = nn.Linear(num_input, hidden1_size)    # connection between input and hidden layer1
        self.output = nn.Linear(hidden1_size, num_classes)
        
        # The model structure can be also defined using "sequential" function
        # self.seq_linear=nn.Sequential(nn.Linear(num_input, hidden1_size),nn.RELU(),nn.Linear(hidden1_size, num_classes))

    # Define "forward" function to perform the computation for input x and return output(s).
    # The function name "forward" is required by Pytorch.
    def forward(self, x):
        # In this implementation, the activation function is reLU, but you can try other functions
        # torch.nn.functional modeule consists of all the activation functions and output functions
        h1_out = AF.relu(self.hidden1(x))
        output = self.output(h1_out)
        # AF.softmax() is NOT needed when CrossEntropyLoss() is used as it already combines both LogSoftMax() and NLLLoss()
        
        # return self.seq_linear(x) # If the model structrue is define by sequential function.
        return output


# To display some images
def show_some_digit_images(images):
    print("> Shapes of image:", images.shape)
    #print("Matrix for one image:")
    #print(images[1][0])
    for i in range(0, 10):
        plt.subplot(2, 5, i+1) # Display each image at i+1 location in 2 rows and 5 columns (total 2*5=10 images)
        plt.imshow(images[i][0], cmap='Oranges') # show ith image from image matrices by color map='Oranges'
    plt.show()

# print confusion matrix with labels
def print_confusion_matrix(cm):
    """
    Prints the confusion matrix with labels for clarity.

    cm: 2x2 numpy array representing the confusion matrix
    """
    print("Confusion Matrix:")
    print("                Predicted")
    print("                 0      1")
    print(f"Actual 0     TN: {cm[0, 0]:<5} FP: {cm[0, 1]:<5}")
    print(f"       1     FN: {cm[1, 0]:<5} TP: {cm[1, 1]:<5}")

# Training function
def train_ANN_model(num_epochs, training_data, device, CUDA_enabled, ANN_model, loss_func, optimizer):
    train_losses = []

    start = time.time()
    ANN_model.train() # to set the model in training mode. Only Dropout and BatchNorm care about this flag.
    end = time.time()
    training_time = end - start

    for epoch_cnt in range(num_epochs):
        for batch_cnt, (features, labels) in enumerate(training_data):
            # Each batch contain batch_size (100) images, each of which 1 channel 28x28
            # print(images.shape) # the shape of images=[100,1,28,28]
            # So, we need to flatten the images into 28*28=784
            # -1 tells NumPy to flatten to 1D (784 pixels as input) for batch_size images
            # the size -1 is inferred from other dimensions
            # images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)

            if (device.type == 'cuda' and CUDA_enabled):
                features = features.to(device) # moving tensors to device
                labels = labels.to(device)

            optimizer.zero_grad() # set the cumulated gradient to zero
            output = ANN_model(features) # feedforward images as input to the network
            loss = loss_func(output, labels) # computing loss
            #print("Loss: ", loss)
            #print("Loss item: ", loss.item())
            train_losses.append(loss.item())
            # PyTorch's Autograd engine (automatic differential (chain rule) package) 
            loss.backward() # calculating gradients backward using Autograd
            optimizer.step() # updating all parameters after every iteration through backpropagation

            # Display the training status
            if (batch_cnt+1) % 10 == 0:
                # print(f"Epoch={epoch_cnt+1}/{num_epochs}, batch={batch_cnt+1}/{num_train_batches}, loss={loss.item()}")
                print(f"Epoch={epoch_cnt+1}/{num_epochs}, batch={batch_cnt+1}, loss={loss.item()}")
    return train_losses, training_time

# Testing function
def test_ANN_model(device, CUDA_enabled, ANN_model, testing_data):
    # torch.no_grad() is a decorator for the step method
    # making "require_grad" false since no need to keeping track of gradients    
    predicted_labels=[]
    true_labels = []
    # torch.no_grad() deactivates Autogra engine (for weight updates). This help run faster
    with torch.no_grad():
        ANN_model.eval() # # set the model in testing mode. Only Dropout and BatchNorm care about this flag.
        for batch_cnt, (features, labels) in enumerate(testing_data):
            # images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)

            if (device.type == 'cuda' and CUDA_enabled):
                features = features.to(device) # moving tensors to device
                labels = labels.to(device)
            
            output = ANN_model(features)
            _, predictions = torch.max(output,1) # returns the max value of all elements in the input tensor
            predicted_labels.extend(predictions.cpu().numpy())
            num_samples = labels.shape[0]
            true_labels.extend(labels.cpu().numpy())
            num_correct = (predictions == labels).sum().item()
            accuracy = num_correct/num_samples
            if (batch_cnt+1) % mini_batch_size == 0:
                print(f"batch={batch_cnt+1}/{num_test_batches}")
        print("> Number of samples =", num_samples, "\nnumber of correct prediction =", num_correct, "\naccuracy =", accuracy)
    return predicted_labels, true_labels

########################### Checking GPU and setup #########################
### CUDA is a parallel computing platform and toolkit developed by NVIDIA. 
# CUDA enables parallelize the computing intensive operations using GPUs.
# In order to use CUDA, your computer needs to have a CUDA supported GPU and install the CUDA Toolkit
# Steps to verify and setup Pytorch and CUDA Toolkit to utilize your GPU in your machine:
# (1) Check if your computer has a compatible GPU at https://developer.nvidia.com/cuda-gpus
# (2) If you have a GPU, continue to the next step, else you can only use CPU and ignore the rest steps.
# (3) Downloaded the compatible Pytorch version and CUDA version, refer to https://pytorch.org/get-started/locally/
# Note: If Pytorch and CUDA versions are not compatible, Pytorch will not be able to recognize your GPU
# (4) The following codes will verify if Pytorch is able to recognize the CUDA Toolkit:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available()):
    print("The CUDA version is", torch.version.cuda)
    # Device configuration: use GPU if available, or use CPU
    cuda_id = torch.cuda.current_device()
    print("ID of the CUDA device:", cuda_id)
    print("The name of the CUDA device:", torch.cuda.get_device_name(cuda_id))
    print("GPU will be utilized for computation.")
else:
    print("CUDA is supported in your machine. Only CPU will be used for computation.")
#exit()

############################### ANN modeling #################################
### Convert the image into numbers: transforms.ToTensor()
# It separate the image into three color channels RGB and converts the pixels of each images to the brightness
# of the color in the range [0,255] that are scaled down to a range [0,1]. The image is now a Torch Tensor (array object)
### Normalize the tensor: transforms.Normalize() normalizes the tensor with mean (0.5) and stdev (0.5)
#+ You can change the mean and stdev values
print("\n------------------ANN modeling---------------------------")
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])
# PyTorch tensors are like NumPy arrays that can run on GPU
# e.g., x = torch.randn(64,100).type(dtype) # need to cast tensor to a CUDA datatype (dtype)

from torch.autograd import Variable
x = Variable

### Download and load the dataset from the torch vision library to the directory specified by root=''
# MNIST is a collection of 7000 handwritten digits (in images) split into 60000 training images and 1000 for testing 
# PyTorch library provides a clean data set. The following command will download training data in directory './data'

############################# MODIFIED ################################
# load dataset
data = pd.read_csv('emails.csv')

# extract features (text) and labels (spam)
texts = data['text']
labels = data['spam']

# convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
features = vectorizer.fit_transform(texts).toarray()

# encode labels (optional is already 0/1)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# split training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=0)

# convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long) # CrossEntropyLoss expects LongTensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# create TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# train_dataset=datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
# test_dataset=datasets.MNIST(root='./data', train=False, transform=transforms, download=False)
# print("> Shape of training data:", train_dataset.data.shape)
# print("> Shape of testing data:", test_dataset.data.shape)
# print("> Classes:", train_dataset.classes)

# You can use random_split function to splite a dataset
#from torch.utils.data.dataset import random_split
#train_data, val_data, test_data = random_split(train_dataset, [60,20,20])

### DataLoader will shuffle the training dataset and load the training and test dataset
mini_batch_size = 100 #+ You can change this mini_batch_size
# If mini_batch_size==100, # of training batches=6000/100=600 batches, each batch contains 100 samples (images, labels)
# DataLoader will load the data set, shuffle it, and partition it into a set of samples specified by mini_batch_size.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=mini_batch_size, shuffle=False)
num_train_batches = len(train_dataloader)
num_test_batches = len(test_dataloader)
print("> Mini batch size: ", mini_batch_size)
print("> Number of batches loaded for training: ", num_train_batches)
print("> Number of batches loaded for testing: ", num_test_batches)

### Let's display some images from the first batch to see what actual digit images look like
iterable_batches = iter(train_dataloader) # making a dataset iterable
images, labels = next(iterable_batches) # If you can call next() again, you get the next batch until no more batch left
show_digit_image = True
# if show_digit_image:
#     show_some_digit_images(images)

### Create an object for the ANN model defined in the MLP class
# Architectural parameters: You can change these parameters except for num_input and num_classes
num_input = X_train.shape[1]   # 28X28=784 pixels of image
num_classes = 2    # output layer
num_neurons_hidden = 128     # number of neurons at the first hidden layer
# Randomly selected neurons by dropout_pr probability will be dropped (zeroed out) for regularization.
dropout_pr = 0.05

# MLP model
MLP_model = MLP(num_input, num_neurons_hidden, num_classes)
# Some model properties: 
# .state_dic(): a dictionary of trainable parameters with their current valeus
# .parameter(): a list of all trainable parameters in the model
# .train() or .eval(): setting training, testing mode

print("> MLP model parameters")
print(MLP_model.parameters)
# state_dict() maps each layer to its parameter tensor.
print ("> MLP model's state dictionary")
for param_tensor in MLP_model.state_dict():
    print(param_tensor, MLP_model.state_dict()[param_tensor].size())

#exit()

# To turn on/off CUDA if I don't want to use it.
CUDA_enabled = True
if (device.type == 'cuda' and CUDA_enabled):
    print("...Modeling using GPU...")
    MLP_model = MLP_model.to(device=device) # sending to whaever device (for GPU acceleration)
else:
    print("...Modeling using CPU...")

### Define a loss function: You can choose other loss functions
class_counts = np.bincount(y_train)
total_samples = len(y_train)
w0 = total_samples / (2 * class_counts[0])
w1 = total_samples / (2 * class_counts[1])
# print(f"w0 = {w0}, w1 = {w1}\n")
weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

loss_func = nn.CrossEntropyLoss(weight=weights)

### Choose a gradient method
# model hyperparameters and gradient methods
# optim.SGD performs gradient descent and update the weigths through backpropagation.
num_epochs = 10
alpha = 0.01       # learning rate
gamma = 0.5        # momentum
# Stochastic Gradient Descent (SGD) is used in this program.
#+ You can choose other gradient methods (Adagrad, adadelta, Adam, etc.) and parameters
MLP_optimizer = optim.SGD(MLP_model.parameters(), lr=alpha, momentum=gamma) 
print("> MLP optimizer's state dictionary")
for var_name in MLP_optimizer.state_dict():
    print(var_name, MLP_optimizer.state_dict()[var_name])

### Train your networks
print("\n............Training MLP................")
train_loss, training_time = train_ANN_model(num_epochs, train_dataloader, device, CUDA_enabled, MLP_model, loss_func, MLP_optimizer)
print(f"\nTraining Time: {training_time} seconds")
print("\n............Testing MLP model................")
print("\n> Input labels:")
print(labels)
predicted_labels, true_labels = test_ANN_model(device, CUDA_enabled, MLP_model, test_dataloader)

print("\n................Performance Results...................")
print(f"Parameters:")
print(f"\tMini batch size: {mini_batch_size}")
print(f"\tNumber of batches loaded for training: {num_train_batches}")
print(f"\tNumber of batches loaded for testing: {num_test_batches}\n")
print(f"\tActivation Function: ReLU")
print(f"\tLoss Function: Cross Entropy Loss")
print(f"\tClass Weights: {weights}")
print(f"\tNumber of classes: {num_classes}")
print(f"\tNeurons: {num_neurons_hidden}")
print(f"\tEpochs: {num_epochs}")
print(f"\tLearning Rate: {alpha}")
print(f"\tGamma (momentum): {gamma}")

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, zero_division=1)
cm = confusion_matrix(true_labels, predicted_labels)
cr = classification_report(true_labels, predicted_labels)

print(f"\nAccuracy: {accuracy}")
print(f"Precision: {precision}")
print_confusion_matrix(cm)
print(f"\nClassification Report:\n{cr}")

print("\n> Predicted labels by MLP model")
print(predicted_labels)
print("> True labels by MLP model")
print(true_labels)

#### To save and load models and model's parameters ####
# To save and load model parameters
#print("...Saving and loading model states and model parameters...")
#torch.save(MLP_model.state_dict(), 'MLP_model_state_dict.pt')
#loaded_MLP_model=MLP(num_input, num_hidden, num_classes)
#loaded_MLP_model=MLP_model.load_state_dict(torch.load('MLP_model_state_dict.pt'))
#torch.save(MLP_optimizer.state_dict(), 'MLP_optimizer_state_dict.pt')
#loaded_MLP_optimizer = MLP_optimizer.load_state_dict(torch.load('MLP_optimizer_state_dict.pt'))

# To save and load a model
#print("...Saving model...")
#torch.save(MLP_model, 'MLP_model_NNIST.pt')
#pretrained_model = torch.load('MLP_model_NNIST.pt')
