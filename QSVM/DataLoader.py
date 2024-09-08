
import numpy as np

def load_data(data_file,N,validation_pts):
    dataset = np.loadtxt('./data/{}.csv'.format(data_file), delimiter=',')

    data = dataset[:N+validation_pts, :-1]
    t = dataset[:N + validation_pts, -1]

    return data,t