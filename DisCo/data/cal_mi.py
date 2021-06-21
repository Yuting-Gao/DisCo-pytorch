import numpy as np


def cal_mi_epoch(model,dataloader):
    t_total = []
    y_total = []

    for x, y in dataloader:
        model.eval()
        t = model(x) ## output before softmax
        t_total.append(t.data.cpu().numpy())
        y_total.append(y)

    y_total = np.concatenate(y_total)
    t_total = np.concatenate(t_total,axis=0)
    num_sample = y_total.shape[0] # should be bigger than NUM_INTEVRAL
    label_matrix = np.zeros((num_sample,10)) # For cifar10, there are total 10 labels
    label_matrix[np.arange(50000) , y_total] = 1

    return label_matrix, t_total
