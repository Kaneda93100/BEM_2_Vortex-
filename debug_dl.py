from src import data_loader as dl
import pathlib
import src.models as mlp
import torch
from torch import nn, optim


path_bem = pathlib.Path('BEM_data_YAW_on.xlsx')
path_sven = pathlib.Path('dataset_forces_mexico.xlsx')
 
S = 'G'
res = 1
inter = 'f'
device = 'cuda'

df = dl.load_clean_data(path_bem, path_sven)
train_F, val_F = dl.get_splits(df, S)
feat_T, label_T = dl.format_data(train_F,S, res, inter = inter, is_train = True, device = device) 
train_dataloader, val_dataloader = dl.MB_build(df, batch_size = 10, entree = S, 
                                               res = res, inter = inter,
                                               device = device)


dummy_machine = mlp.TurbineMLP(2, 2, 3, 5, 0.2, device = device)
opt = optim.Adam(dummy_machine.parameters(), lr = 1e-4, weight_decay=1)
loss_func = nn.MSELoss(reduction = 'mean')
for ep in range(10) : 
    
    dummy_machine.train()
    for feat, lab in train_dataloader :
        opt.zero_grad()
        loss = loss_func(input = dummy_machine(feat), target=lab)
        loss.backward()        
    
    dummy_machine.eval()
    val_err = []
    for feat, lab in val_dataloader :
        val_err.append(loss_func(input = dummy_machine(feat), target = lab))
