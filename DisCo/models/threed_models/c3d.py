import torch
import torch.nn as nn

'''
https://github.com/jfzhang95/pytorch-video-recognition/blob/master/network/C3D_model.py
'''


__all__ = ['c3d']


defaultcfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    '11pruned_aa' : [64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256]
}

class C3D(nn.Module):
    """
    The C3D network.
    """
    def __init__(self, num_classes=1000, dropout_ratio=0.2, without_t_stride=False,
                 pooling_method='max', cfg='11', pretrained=''):
        super(C3D, self).__init__()

        if cfg == '11':
            self.cfg = cfg = defaultcfg['11']
            self.model_path = pretrained
            # '/data/home/jiaxzhuang/.cache/torch/checkpoints/c3d-pretrained.pth'
            is_pretrained = True
        else:
            self.cfg = cfg = defaultcfg[cfg]
            is_pretrained = False
            if pretrained:
                self.model_path = pretrained
                is_pretrained = True

        self.pooling_method = pooling_method.lower()
        self.without_t_stride = without_t_stride

        self.conv1 = nn.Conv3d(3, cfg[0], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(cfg[0], cfg[2], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(cfg[2], cfg[4], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(cfg[4], cfg[5], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(cfg[5], cfg[7], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(cfg[7], cfg[8], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(cfg[8], cfg[10], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(cfg[10], cfg[11], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(cfg[-1]*16, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=dropout_ratio)

        self.relu = nn.ReLU()

        self.__init_weight()


        if is_pretrained:
            print('=> Using pretrained.')
            self.__load_pretrained_weights()



    def mean(self, modality='rgb'):
        # return [0.5, 0.5, 0.5] if modality == 'rgb' else [0.5]
        return [0.398, 0.38, 0.35] if modality == 'rgb' else [0.5]


    def std(self, modality='rgb'):
        # return [0.5, 0.5, 0.5] if modality == 'rgb' else [0.5]
        return [1.0, 1.0, 1.0] if modality == 'rgb' else [0.5]

    @property
    def network_name(self):
        name = 'c3d'
        if not self.without_t_stride:
            name += "-ts-{}".format(self.pooling_method)
        return name

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, self.cfg[-1]*16)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }


        p_dict = torch.load(self.model_path)
        print('=> Load pretrained from: ', self.model_path)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


def c3d(num_classes, dropout, without_t_stride, pooling_method, cfg=None, pretrained='', **kwargs):
    model = C3D(num_classes=num_classes, dropout_ratio=dropout,
                without_t_stride=without_t_stride, pooling_method=pooling_method, cfg=cfg, pretrained=pretrained)
    # new_model_state_dict = model.state_dict()
    # state_dict = model_zoo.load_url(model_urls['googlenet'], map_location='cpu', progress=True)
    # state_d = inflate_from_2d_model(state_dict, new_model_state_dict,
    #                                 skipped_keys=['fc', 'aux1', 'aux2'])
    # model.load_state_dict(state_d, strict=False)
    return model

if __name__ == "__main__":
    inputs = torch.rand(40, 3, 16, 112, 112)
    # net = c3d(num_classes=101, dropout=0.5, without_t_stride=False, pooling_method='max')

    cfg = '11pruned_aa'
    net = c3d(num_classes=101, dropout=0.5, without_t_stride=False, pooling_method='max', cfg=cfg)

    print(net)
    outputs = net.forward(inputs)
    print(outputs.size())
