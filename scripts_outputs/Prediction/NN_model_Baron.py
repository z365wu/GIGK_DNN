import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
#import pickle as pkl

import torch.nn.functional as F
#import torch.optim as optim
import pandas as pd


def Baron2024(moments_Arrival, moments_Service):
    
    num_moms_arrive = 5
    num_moms_service = 5
    
    ## Insert your inter-arrival and service time moments here:

    ### Example:

    #inter_arrival_moms = torch.tensor([3.7960, 9.3644, 36.2736, 192.3857, 1290.4828])
    #service_moments = torch.tensor([3, 5.9023, 17.2487, 67.1530, 327.2156])

    m = nn.Softmax(dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # code made in pytorch3.ipynb with comments
    class Net(nn.Module):

        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(num_moms_arrive + num_moms_service - 1, 50)
            self.fc2 = nn.Linear(50, 70)
            self.fc3 = nn.Linear(70, 100)
            self.fc4 = nn.Linear(100, 150)
            self.fc5 = nn.Linear(150, 200)
            self.fc6 = nn.Linear(200, 200)
            self.fc7 = nn.Linear(200, 350)
            self.fc8 = nn.Linear(350, 499)

        def forward(self, x):

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))
            x = self.fc8(x)
            return x

    net = Net().to(device)
    #net.load_state_dict(torch.load(os.path.join(args.model_path, file), map_location=torch.device('cpu')))
    net.load_state_dict(torch.load('Output/models/pytorch_g_g_1_opher2023.pkl', map_location=torch.device('cpu')))
    
    # Original name of 'pytorch_g_g_1_opher2023.pkl' is
    # pytorch_g_g_1_true_moms_new_data_archi_3_bs_128_weight_decay_5_num_moms_arrival_5_num_moms_service_5_lr_first_0.75_lr_second_1.0_19_55_55_283731.pkl
    
    # get the first 5 moments
    inter_arrival_moms = moments_Arrival[:num_moms_arrive]
    service_moments = moments_Service[:num_moms_service]
    

    #inter_arrival_moms = torch.tensor(inter_arrival_moms.tolist())
    #service_moments = torch.tensor(service_moments.tolist())
    inter_arrival_moms = torch.tensor(inter_arrival_moms.tolist())
    service_moments = torch.tensor(service_moments.tolist())
    
    service_1_mom = service_moments[0]
    inter_arrival_moms = inter_arrival_moms / service_1_mom
    service_moments = service_moments / service_1_mom
    input_moms = torch.log(torch.cat((inter_arrival_moms, service_moments[1:]), axis=0))
    input_moms = input_moms.reshape((1, input_moms.shape[0]))

    with torch.no_grad():
        predictions = m(net(input_moms))
        normalizing_const = 1 / torch.exp(input_moms[:, 0])
        predictions = predictions * normalizing_const.reshape((input_moms.shape[0], 1))
        prob_0 = (1 - torch.sum(predictions[:, :], axis=1)).reshape(torch.sum(predictions[:, :], axis=1).shape[0], 1)
        preds = torch.concat((prob_0, predictions), axis=1)


    ## Number of values to present in the starionay queue lenght distribution.
    max_probs = 500
    true = preds[0, :max_probs].tolist()

    return true


