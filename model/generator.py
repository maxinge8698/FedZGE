import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=28, num_classes=10, slope=0.2, conditional=True):
        super(Generator, self).__init__()
        if isinstance(img_size, (list, tuple)):
            self.init_size = (img_size[0] // 4, img_size[1] // 4)
        else:
            self.init_size = (img_size // 4, img_size // 4)
        if conditional:
            self.embedding = nn.Embedding(num_classes, nz)
            self.linear = nn.Sequential(nn.Linear(nz * 2, ngf * 2 * self.init_size[0] * self.init_size[1]))
        else:
            self.linear = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size[0] * self.init_size[1]))

        self.bn0 = nn.BatchNorm2d(ngf * 2)

        self.deconv1 = nn.ConvTranspose2d(ngf * 2, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 2)
        self.relu1 = nn.LeakyReLU(slope, inplace=True)

        self.deconv2 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf)
        self.relu2 = nn.LeakyReLU(slope, inplace=True)

        self.conv3 = nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(nc)
        self.tanh = nn.Tanh()

        # # Initialization
        # for m in self.modules():
        #     if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
        #         nn.init.normal_(m.weight, 0.0, 0.02)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     if isinstance(m, nn.BatchNorm2d):
        #         nn.init.normal_(m.weight, 1.0, 0.02)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, z, y=None):
        if y is not None:
            label_embedding = self.embedding(y)
            h = torch.cat((z, label_embedding), dim=1)
        else:
            h = z
        x = self.linear(h)
        x = x.view(x.shape[0], -1, self.init_size[0], self.init_size[1])

        x = self.bn0(x)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.tanh(x)
        return x


if __name__ == '__main__':
    class args:
        num_classes = 10
        batch_size = 256
        nz = 100
        nc = 3
        img_size = 32


    # Sample random noise: z~N(0,1)
    z = torch.randn((args.batch_size, args.nz)).cuda()
    print(z.shape)
    # Sample one-hot label: y~U(0,C-1)
    y = torch.randint(low=0, high=args.num_classes, size=(args.batch_size,))
    y = y.sort()[0]
    y = y.cuda()
    print(y.shape)

    G = Generator(nz=args.nz, nc=args.nc, img_size=args.img_size, num_classes=args.num_classes, conditional=True).cuda()
    print('Model parameters: %2.2fM' % (sum(p.numel() for p in G.parameters()) / (1000 * 1000)))
    x = G(z, y)  # (250,1,)
    print(x.shape)
