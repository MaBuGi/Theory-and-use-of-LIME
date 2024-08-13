import datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from cnn import Net

#Download MNIST train and test dataset
mnist_train = datasets.MNIST(root='./data', download=True, train=True, 
    transform=transforms.Compose([transforms.ToTensor()]))
    
mnist_testset = datasets.MNIST(root='./data', download=True, train=False,
    transform=transforms.Compose([transforms.ToTensor()]))
    
train_loader = torch.utils.data.DataLoader(
                    mnist_train, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(
                    mnist_testset, batch_size=1, shuffle=True)
#Init net, optimizer, criterion
net = Net()
optimizer = torch.optim.Adam(net.parameters(), 0.001)
criterion = nn.CrossEntropyLoss()
#Train loop on training dataset
for j in range(2):
  for batch, (inputs, labels) in enumerate(train_loader):
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(batch % 200 == 0):
      print(f"Epoch: {j} at loss: {loss}")
      
#Evaluate on testset
def evaluate(net, mnist_testset):
  net.eval()
  error = 0
  for i, (input, label) in enumerate(test_loader):
    if torch.argmax(net(input)).item() != label.item():
      error += 1
  return error/len(test_loader)
print(f"Errorrate: {evaluate(net, mnist_testset)}")
