import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    


class SimpleDeepNN(nn.Module):
    def __init__(self, input_size=2048, output_size=2, dropout_prob=0.5):
        super(SimpleDeepNN, self).__init__() # load from parent
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

        # relu and drop out will be reused
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)  # Apply dropout with specified probability

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x) # softmax activation is applied in loss function
        return x
    

class PyramidFullDropOut(nn.Module):
    def __init__(self, input_size=2048, output_size=2, dropout_prob=0.5):
        super(PyramidFullDropOut, self).__init__() # load from parent
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output_size)

        # relu and drop out will be reused
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)  # Apply dropout with specified probability

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x) # softmax activation is applied in loss function
        return x
    

class PyramidEndDropout(nn.Module):
    def __init__(self, input_size=2048, output_size=2, dropout_prob=0.5):
        super(PyramidEndDropout, self).__init__() # load from parent
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output_size)

        # relu and drop out will be reused
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)  # Apply dropout with specified probability

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x) # softmax activation is applied in loss function
        return x
    

class PyramidEndDropout(nn.Module):
    def __init__(self, input_size=2048, output_size=2, dropout_prob=0.5):
        super(PyramidEndDropout, self).__init__() # load from parent
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output_size)

        # relu and drop out will be reused
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)  # Apply dropout with specified probability

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x) # softmax activation is applied in loss function
        return x
    

class RectangleFullDropout(nn.Module):
    def __init__(self, input_size=2048, hidden_size=1024, output_size=2, dropout_prob=0.5):
        super(RectangleFullDropout, self).__init__() # load from parent
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)

        # relu and drop out will be reused
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)  # Apply dropout with specified probability


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x) # softmax activation is applied in loss function
        return x


class RectangleEndDropout(nn.Module):
    def __init__(self, input_size=2048, hidden_size=1024, output_size=2, dropout_prob=0.5):
        super(RectangleEndDropout, self).__init__() # load from parent
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)

        # relu and drop out will be reused
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)  # Apply dropout with specified probability


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x) # softmax activation is applied in loss function
        return x
    


class SmallRectangleEndDropout(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, output_size=2, dropout_prob=0.5):
        super(SmallRectangleEndDropout, self).__init__() # load from parent
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)

        # relu and drop out will be reused
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)  # Apply dropout with specified probability


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x) # softmax activation is applied in loss function
        return x
    


class ReallySmallRectangleEndDropout(nn.Module):
    def __init__(self, input_size=2048, hidden_size=128, output_size=2, dropout_prob=0.5):
        super(ReallySmallRectangleEndDropout, self).__init__() # load from parent
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)

        # relu and drop out will be reused
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)  # Apply dropout with specified probability


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x) # softmax activation is applied in loss function
        return x






