from math import sqrt
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  def __init__(self, num_classes=10):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 5, padding = 2, stride = 2)
    self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = 5, padding = 2, stride = 2)
    self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 8, kernel_size = 5, padding = 2, stride = 2)
    self.fc_1 = nn.Linear(32, 128)
    self.fc_2 = nn.Linear(128, num_classes)
    self.dropout_fc = nn.Dropout(p=0.5)
    self.batch_norm_1 = nn.BatchNorm2d(16)
    self.batch_norm_2 = nn.BatchNorm2d(64)
    self.batch_norm_3 = nn.BatchNorm2d(8)


    self.init_weights()

  def init_weights(self):
    torch.manual_seed(42)

    for conv in [self.conv1, self.conv2, self.conv3]:
        C_in = conv.weight.size(1)
        nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
        nn.init.constant_(conv.bias, 0.0)

    D_in = self.fc_1.weight.size(1)
    nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(D_in))
    nn.init.constant_(self.fc_1.bias, 0.0)

    D_in = self.fc_2.weight.size(1)
    nn.init.normal_(self.fc_2.weight, 0.0, 1 / sqrt(D_in))
    nn.init.constant_(self.fc_2.bias, 0.0)

  def forward(self, x):
        N, C, H, W = x.shape

        x = F.relu(self.batch_norm_1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm_2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm_3(self.conv3(x)))
        x = x.view(N, -1)
        x = self.dropout_fc(F.relu(self.fc_1(x)))
        x = self.dropout_fc(self.fc_2(x))
        return x
