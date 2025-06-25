import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrimaryCaps(nn.Module):
    # 理清一下思路，作为一个 layer 的话，他处理的其实就是输入与输出的关系
    # 所以这里要处理的就是，如何得到 primarycaps 的输出
    # 要拿到输入的维度，是 256 个 channel
    # 输出的形式是 32 * 6 * 6 的 tensor 了

    def __init__(
        self,
        cap_dim = 8,
        in_channels = 256,
        out_channels = 32,
        kernel_size = 9,
        ):
        super(PrimaryCaps, self).__init__()

        self.cap_dim = cap_dim
        self.out_channels = out_channels

        # it should be noted that the out_channels is the number of capsules
        # because each capsule should have its own convolutional layer with the output dimension of cap_dim
        self.CapConv = nn.ModuleList([
            nn.Conv2d(in_channels, cap_dim, kernel_size=kernel_size, stride=2)
            for _ in range(out_channels)
        ])

    
    def forward(self, x):
        u = [self.CapConv[i](x) for i in range(self.out_channels)] # [ out_channels, (b, c, h, w)]
        u = torch.stack(u, dim=1)  # (batch_size, out_channels, cap_dim, h, w)
        u = u.permute(0, 1, 3, 4, 2)  # (batch_size, out_channels, h, w, cap_dim)

        return self.squash(u)

    def squash(self, x):
        # Squash function
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        output_tensor = squared_norm * x / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):

    def __init__(
        self,
        in_caps = 8,
        in_channels = 32, 
        out_caps = 16,
        out_channels = 10,
        routing_iterations = 3,
        ):

        super(DigitCaps, self).__init__()

        self.in_caps = in_caps
        self.in_channels = in_channels
        self.out_caps = out_caps
        self.out_channels = out_channels
        self.routing_iterations = routing_iterations

        # share weigths between same types of capsules, which is represented by the in_channels
        self.W = nn.Parameter(torch.randn(1, 1, in_channels, out_channels, out_caps, in_caps))

    def forward(self, x):
        # first calculate the vote capsules, which should have the same dimension as the output capsules
        u = x.permute(0, 2, 3, 1, 4) # (batch_size, h, w, in_channels, in_caps)
        u = u.unsqueeze(-2).unsqueeze(-1)

        u_hat = torch.matmul(self.W, u)  # → shape: (B, 6, 6, 32, 10, 16, 1)
        u_hat = u_hat.squeeze(-1)                 # → (B, 6, 6, 32, 10, 16)
        u_hat = u_hat.view(x.size(0), -1, self.out_channels, self.out_caps)

        b_ij = torch.zeros(*u_hat.size()[:-1]).to(device)

        v_j = self.dynamic_routing(b_ij, u_hat, routing_iterations=self.routing_iterations)

        return v_j

    def dynamic_routing(self, b_ij, u_hat, routing_iterations):

        for i in range(routing_iterations):
            c_ij = F.softmax(b_ij, dim = -1)
            c_ij = c_ij.unsqueeze(-1)           # add a dimension for boardcasting
            s_j = (c_ij * u_hat).sum(dim=1)     # → shape: (B, 10, 16)
            v_j = self.squash(s_j)
            
            if i < self.routing_iterations - 1:
                v_j = v_j.unsqueeze(1)
                a_j = (u_hat * v_j).sum(dim=-1)  
                b_ij = b_ij + a_j # → shape: (B, 32, 10)
        return v_j

    def squash(self, x):
        # Squash function
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        output_tensor = squared_norm * x / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class CapsuleNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(CapsuleNet, self).__init__()
        self.num_classes = num_classes

        self.Conv1 = nn.Conv2d(1, 256, kernel_size = 9, stride = 1)
        self.relu = nn.ReLU()
        self.primary_caps = PrimaryCaps(
            cap_dim=8,
            in_channels = 256,
            out_channels = 32,
            kernel_size = 9,
        )

        self.digit_caps = DigitCaps(
            in_caps = 8,
            in_channels = 32,
            out_caps = 16,
            out_channels = num_classes,
            routing_iterations = 3
        )

    def forward(self, x):

        x_out = self.relu(self.Conv1(x))
        x_out = self.primary_caps(x_out)
        x_out = self.digit_caps(x_out)

        return x_out

    # def margin_loss(self, x, label, m_pos=0.9, m_neg=0.1, lambda_=0.5):
    #     batch_size = x.size(0)
        
    #     # x: (B, 10, 16)
    #     v_k = torch.sqrt((x**2).sum(dim=2))  # → (B, 10)


    #     one_hot = F.one_hot(label, num_classes=self.num_classes).float()

    #     pos = F.relu(m_pos - v_k).pow(2)
    #     neg = F.relu(v_k - m_neg).pow(2)

    #     loss = one_hot * pos + lambda_ * (1.0 - one_hot) * neg
    #     return loss.sum(dim=1).mean()


    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss


















