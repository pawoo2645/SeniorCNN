#import torch

import torch as th

#device = th.device("cuda:0")

th.cuda.is_available()

#model.to(device)


print ('Available devices ', th.cuda.device_count())
#print ('Available devices ', device)

# print ('Current cuda device ', th.cuda.current_device())

# torch.cuda.get_device_name(0)
#
# torch.cuda.is_available()
#
# import torch
# print(torch.__version__)