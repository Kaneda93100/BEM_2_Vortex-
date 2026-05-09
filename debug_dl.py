from src import data_loader_G as dl
import pathlib

path_bem = pathlib.Path('BEM_data_YAW_on.xlsx')
path_sven = pathlib.Path('dataset_forces_mexico.xlsx')
 
S = 'G'
res = 0
inter = 'f'

df = dl.load_clean_data(path_bem, path_sven)
train_F, val_F = dl.get_splits(df, S)
feat_T, label_T = dl.format_data(train_F,S, res, inter = inter, is_train = True) 
train_dataloader, val_dataloader = dl.MB_build(df, batch_size = 10, entree = S, 
                                               res = res, inter = inter)
x = 0