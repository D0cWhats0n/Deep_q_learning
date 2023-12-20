import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, input_shape, num_actions, architecture=(16, 16, 32, 64)):
        super(ConvNet, self).__init__()
        self.conv_layers = self.get_conv_layers(input_shape[0], architecture)

        conv_out_size = self.get_conv_layer_out_size(input_shape, architecture)
        self.linear1 = nn.Linear(conv_out_size, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 64)
        # Output of neural net gives Q values for each action, hence output_dim = num_actions  
        self.out = nn.Linear(64, num_actions)
        self.cuda()

    def get_conv_layers(self, chans, architecture):
        conv_layer_list = []
        in_chan = chans
        for out_chan in architecture:
            conv_layer_list.append(nn.Conv2d(in_chan, out_chan, 3, padding="same"))
            conv_layer_list.append(nn.ReLU())
            conv_layer_list.append(nn.BatchNorm2d(out_chan))
            conv_layer_list.append(nn.MaxPool2d(2))
            in_chan = out_chan
        conv_layer_list.append(nn.Flatten())
        return nn.ModuleList(conv_layer_list)
    
    def get_conv_layer_out_size(self, input_shape, architecture):
        '''Output size of conv_layers is given by number of max pooling operations 
        and last layer channel size. Direct calculation is hard because of rounding errors for 
        uneven feature map sizes.
        '''
        x_shape = input_shape[1]
        y_shape = input_shape[2]
        for _ in architecture:
            x_shape = x_shape//2
            y_shape = y_shape//2
        return x_shape * y_shape * architecture[-1]


    def forward(self, x):
        for el in self.conv_layers:
            x = el(x)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))

        return self.out(x)


if __name__ =="__main__":
    import numpy as np
    from torchsummary import summary
    num_actions = 3
    input_shape = (20, 3, 64, 64)
    conv_net = ConvNet((input_shape[1],input_shape[2],input_shape[3]), num_actions)
    x = np.ones(input_shape)
    x = torch.tensor(x).float()
    x = x.to("cuda")
    out = conv_net(x)
    summary(conv_net, (3,64,64))