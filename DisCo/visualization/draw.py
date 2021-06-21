from time import time
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE


def get_embd(embd_path, label_path):
    num = 50
    with open(embd_path, 'rb') as data:
        data_embd = np.load(data)

    with open(label_path, 'rb') as data:
        labels = np.load(data)

    sel_labels = []
    sel_embds = []
    for class_index in range(5, 15):
        sel_embd = data_embd[labels == class_index][: num]
        sel_embds.extend(sel_embd)
        sel_labels.extend([class_index]*num)
#     data_embd = []
#     labels = []
#     with torch.no_grad():
#         for data, label in loader:
#             data = data.to(device)
#             embd = model(data)
#             print(embd.size())
#             data_embd.append(embd.detach().cpu().numpy())
#             labels.append(label)
    # print(len(sel_embds))
    # print(len(sel_labels))
    n_features = len(labels)
    return sel_embds, sel_labels, n_features


def plot_embedding(result, label, title):
    x_min, x_max = np.min(result, 0), np.max(result, 0)
    data = (result - x_min) / (x_max - x_min)
    plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.title(title)
    plt.show()
    plt.savefig('tsn.png')


def main():
    from sklearn.manifold import TSNE
    import seaborn as sns
    import pandas as pd
    sns.set_style("whitegrid")
    # data, label, n_features = get_embd(embd_path, label_path)
    # embd_paths = ['./data/mocov2_r18/resnet18_embd',
    #               './data/r50_r18/resnet18_embd',
    #               './data/r50/resnet50_embd',
    #               './data/effib0/efficientb0_embd',
    #               './data/r50_effib0/efficientb0_embd',
    #               './data/r50w2/resnet50w2_embd',
    #               './data/r50w2_r18/resnet18_embd'
    #               './data/r101/resnet101_embd',
    #               './data/r34/resnet34_embd']
    # label_paths = ['./data/mocov2_r18/resnet18_label',
    #                './data/r50_r18/resnet18_label',
    #                './data/r50/resnet50_label',
    #                './data/effib0/efficientb0_label',
    #                './data/r50_effib0/efficientb0_label',
    #                './data/r50w2/resnet50w2_label',
    #                './data/r50w2_r18/resnet18_label',
    #                './data/r101/resnet101_label',
    #                './data/r34/resnet34_label']

    embd_paths = ['./data/r101_r34/resnet34_embd']
    label_paths = ['./data/r101_r34/resnet34_label']
    for embd_path, label_path in zip(embd_paths, label_paths):
        data, label, n_features = get_embd(embd_path, label_path)
        name = embd_path.split('/')[-2]
        # from cuml.manifold import TSNE
        # result = tsne.fit_transform(data)
        # tsne = TSNE(n_components = 2)
        t0 = time()
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(data)
        tsne_data = np.vstack((result.T, label)).T
        tsne_df = pd.DataFrame(data=tsne_data,
                               columns=('Dim_1', 'Dim_2', 'label'))

        sns.FacetGrid(tsne_df, hue='label', size=6).map(plt.scatter, 'Dim_1',
                                                        'Dim_2').add_legend()
        plt.savefig('./pdf/' + name + '.pdf')
        print('{}: {}'.format(name, time() - t0))


if __name__ == '__main__':
    main()
