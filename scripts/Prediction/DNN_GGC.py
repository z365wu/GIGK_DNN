import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
#import pickle as pkl

import torch.nn.functional as F
#import torch.optim as optim
import pandas as pd
from Sampling.Save_Combine_Read_for_CSV_files import Load_Samples_from_file


def GGC(moments_Arrival, moments_Service, K):
    
    num_arrival_moms, num_ser_moms = 5, 5
    
    ## Insert your inter-arrival and service time moments here:

    ### Example:

    #inter_arrival_moms = torch.tensor([3.7960, 9.3644, 36.2736, 192.3857, 1290.4828])
    #service_moments = torch.tensor([3, 5.9023, 17.2487, 67.1530, 327.2156])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Net(nn.Module):
    
        def __init__(self, input_size, output_size):
            
            super().__init__()
            
            self.fc1 = nn.Linear(input_size , 50)
            self.fc2 = nn.Linear(50, 70)
            self.fc3 = nn.Linear(70, 200)
            self.fc4 = nn.Linear(200, 350)
            self.fc5 = nn.Linear(350, 600) # changed from 600
            self.fc6 = nn.Linear(600, output_size)
    
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = self.fc6(x)
            return x  

    m = nn.Softmax(dim=1)
    input_size = 11
    output_size = 500
    net = Net(input_size, output_size).to(device)
    # net.load_state_dict(torch.load('./models/num_moms_5_layer_6.pkl', map_location=torch.device('cpu')))
    net.load_state_dict(torch.load('Prediction/models/num_moms_5_layer_6.pkl', map_location=torch.device('cpu')))
           
    
    # get the first 5 moments
    inter_arrival_moms = moments_Arrival[:num_arrival_moms]
    service_moments = moments_Service[:num_ser_moms]
    print('inter_arrival_moms', inter_arrival_moms)

    #inter_arrival_moms = torch.tensor(inter_arrival_moms.tolist())
    #service_moments = torch.tensor(service_moments.tolist())
    inter_arrival_moms = torch.tensor(inter_arrival_moms.tolist())
    service_moments = torch.tensor(service_moments.tolist())
    
    
    #arr_1_mom = inter_arrival_moms[0]
    #inter_arrival_moms = inter_arrival_moms / arr_1_mom
    input_moms = torch.cat((inter_arrival_moms, service_moments), axis=0)
    # log transform
    input_moms = torch.log(input_moms)
    # insert the number of servers K
    input_moms = torch.cat((input_moms, torch.tensor([K]))).to(torch.float64)
    input_moms = input_moms.reshape((1, input_moms.shape[0]))
    print(input_moms)

    with torch.no_grad():
        predictions = m(net(input_moms.to(device).float()))

    ## Number of values to present in the starionay queue lenght distribution.
    max_probs = 500
    true = predictions[0, :max_probs].tolist()
    # print([true[x]- stationary_1[x] for x in range(0,10)])
    
    return true


# if __name__ == '__main__':
    
#     ## test
#     # data sample
#     K=2
#     file_name = f'../Output/samples/continuous/K{K}/df_continuous_{K}_servers_sample_sd_50.csv'
#     data1 = pd.read_csv(file_name)
    
#     Moments, stationary, _ = Load_Samples_from_file(10, 500, data1)
#     moments_Arrival, moments_Service = Moments[8,:10], Moments[8,10:]
    
#     # exponential transfer to back real moments
#     moments_Arrival = np.exp(moments_Arrival) - 1
#     moments_Service = np.exp(moments_Service) - 1
    
#     stationary_1 = stationary[8][:20]
    
#     Prob_dist = GGC(moments_Arrival, moments_Service, K)
