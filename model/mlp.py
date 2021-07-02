import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_size,
        h_sizes,
        out_size,
        in_drop_p,
        drop_ps,
        have_last_bn=False,
        pretrained_model_path="",
    ):
        super(MLP, self).__init__()

        self.in_drop = nn.Dropout(in_drop_p)

        linears = [nn.Linear(in_size, h_sizes[0])]
        drops = [nn.Dropout(drop_ps[0])]
        for i, hs in enumerate(h_sizes[1:], start=1):
            linears.append(nn.Linear(h_sizes[i - 1], hs))
            drops.append(nn.Dropout(p=drop_ps[i]))

        self.fc_layers = nn.ModuleList(linears)
        self.drop_layers = nn.ModuleList(drops)

        self.fc_out = nn.Linear(h_sizes[-1], out_size)

        self.have_last_bn = have_last_bn
        if have_last_bn:
            self.bn = nn.BatchNorm1d(out_size)

    #     self.__init_layers()

    # def __init_layers(self):
    #     for m in self.modules():
    #         if type(m) == nn.Linear:
    #             nn.init.xavier_normal_(m.weight)
    #             if not m.bias is None:
    #                 nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.in_drop(x)
        for fc, drop in zip(self.fc_layers, self.drop_layers):
            h = drop(torch.relu(fc(h)))
        h = self.fc_out(h)

        if self.have_last_bn:
            h = self.bn(h)

        return h
