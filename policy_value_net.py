import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

DEVICE = torch.device('cuda')
LR = 1e-3
L2_WEIGHT_DECAY = 1e-4

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ResidualBlock(nn.Module):
    def __init__(self, n_f):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
        nn.Conv2d(n_f, n_f, 3, 1, 1), 
        nn.BatchNorm2d(n_f),
        nn.ReLU(),
        nn.Conv2d(n_f, n_f, 3, 1, 1),
        nn.BatchNorm2d(n_f),
        )

    def forward(self, x):
        x = x + self.residual(x)
        x = F.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, board_size, n_f=256, n_res=3):
        super(Net, self).__init__()

        common_module_lst = nn.ModuleList([
        nn.Conv2d(4, n_f, 3, 1, 1),
        nn.BatchNorm2d(n_f),
        nn.ReLU()
        ])
        common_module_lst.extend([ResidualBlock(n_f) for _ in range(n_res)])
        self.body = nn.Sequential(*common_module_lst)

        self.head_p = nn.Sequential(
            nn.Conv2d(n_f, 2, 1, 1),  
            nn.BatchNorm2d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size),
            nn.LogSoftmax(dim=-1)
        )

        self.head_v = nn.Sequential(
            nn.Conv2d(n_f, 1, 1, 1),  
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(board_size * board_size, 1),
            nn.Tanh()
        )
        self.to(DEVICE)

    def forward(self, x):
        x = self.body(x)
        p = self.head_p(x)
        v = self.head_v(x)
        return p, v

class PolicyValueNet:

    def __init__(self, board_width, board_height, model_file=None,
                    init_lr=LR, weight_decay=L2_WEIGHT_DECAY,use_gpu=True):
        self.board_width = board_width
        self.board_height = board_height
        board_size = min(board_width, board_height) # use square board
        self.use_gpu = use_gpu
        if use_gpu:
            self.policy_value_net = Net(board_size).to(DEVICE)
        else:
            self.policy_value_net = Net(board_size)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
        lr=init_lr, betas=[0.7, 0.99],
        weight_decay=weight_decay)
        self.l2_loss = nn.MSELoss()
        if model_file:
            self.restore_model(model_file)

    def policy_value(self, state_batch):
        state_batch = Variable(torch.FloatTensor(state_batch).cuda())
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    def policy_value_fn(self, board):
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        state_batch = Variable(torch.FloatTensor(state_batch).cuda())
        mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
        winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        log_act_probs, value = self.policy_value_net(state_batch)

        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
                
        return loss.item(), entropy.item()
    
    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_path):
        torch.save(self.policy_value_net.state_dict(), model_path)

    def restore_model(self, model_path):
        self.policy_value_net.load_state_dict(torch.load(model_path))