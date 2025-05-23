import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinModel(nn.Module):
    def __init__(self, input_dim=23, embed_dim=128, lstm_output_size=80, num_classes=1):
        super(ProteinModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(input_dim, embed_dim)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(embed_dim, 10, 10)
        self.conv2 = nn.Conv1d(10, 10, 8)
        self.conv3 = nn.Conv1d(10, 10, 5)
        
        # MaxPooling layers
        self.pool = nn.MaxPool1d(2)
        
        # LSTM layer
        self.lstm = nn.LSTM(10, lstm_output_size, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(lstm_output_size * 2, num_classes)
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        # Convolution and pooling layers
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, embed_dim, sentence_length)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Prepare for LSTM layer
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, seq_length, num_filters)
        
        # LSTM layer
        x, (hn, cn) = self.lstm(x)
        
        # Concatenate LSTM output (hn) from both proteins (simulate by doubling the dimension)
        x = torch.cat((hn[-1], hn[-1]), dim=1)  # Assume hn is the same for both proteins
        
        
        return x
    
class DNN(nn.Module):
    def __init__(self,):
        super().__init__()
        self.pm = ProteinModel()
        self.linear = nn.Linear(320,1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x1,x2):
        # Embedding
        x1 = x1.cuda()
        x2 = x2.cuda()

        x1 = self.pm(x1)
        x2 = self.pm(x2)

        concatenated_tensor = torch.cat((x1, x2), dim=1)

        return self.sigmoid(self.linear(concatenated_tensor))


if __name__=="__main__":

    model = DNN()
    input_data1 = torch.randint(0, 23, (128, 150))  # Example input tensor
    input_data2 = torch.randint(0, 23, (128, 150))

    output = model(input_data1,input_data2)
    print(output.shape)
