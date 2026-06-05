import torch
from torch import nn
import torch.nn.functional as F

def conv_2_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

class VGG16(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight) # 원본
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*20, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x

class VGG16_xavier(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_xavier, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*20, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x

class VGG16_kaiming(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_kaiming, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*20, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x
    
class VGG16_constant_init(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.constant_(module.weight, 1e-2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.constant_(module.weight, 1e-2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_constant_init, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*20, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x

class VGG16_gaussian_init_std1(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0, std=1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_gaussian_init_std1, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*20, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x

class VGG16_gaussian_init_std1e_3(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=1e-3)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0, std=1e-3)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_gaussian_init_std1e_3, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*20, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x



class VGG16_depth4(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_depth4, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            # conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim* 15*41, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x

class VGG16_depth4_fc2(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_depth4_fc2, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            # conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim* 15*41, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x

class VGG16_depth4_fc1(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_depth4_fc1, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            # conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim* 15*41, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x



class VGG16_depth3(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_depth3, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            # conv_3_block(4*base_dim,8*base_dim), #512
            # conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(4*base_dim* 31*83, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x

class VGG16_depth3_fc2(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_depth3_fc2, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            # conv_3_block(4*base_dim,8*base_dim), #512
            # conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(4*base_dim* 31*83, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x

class VGG16_depth3_fc1(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_depth3_fc1, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            # conv_3_block(4*base_dim,8*base_dim), #512
            # conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(4*base_dim* 31*83, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x


class VGG16_depth2(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_depth2, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            # conv_3_block(2*base_dim,4*base_dim), #256
            # conv_3_block(4*base_dim,8*base_dim), #512
            # conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(2*base_dim* 62*167, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape) # b 512 7 20
        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x


class VGG16_depth4_fc1_concat(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_depth4_fc1_concat, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            # conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(120* 125*335, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        f1 = self.feature[0](x)
        f2 = self.feature[0:2](x)
        f3 = self.feature[0:3](x)
        f4 = self.feature[0:4](x)

        b, c, h, w = f1.shape
        up_f2 = self.resizing(f2, height=h, width=w)
        up_f3 = self.resizing(f3, height=h, width=w)
        up_f4 = self.resizing(f4, height=h, width=w)
        concat_f = torch.cat([f1, up_f2, up_f3, up_f4], dim=1)
        x = concat_f

        # x = self.feature(x)
        # print(x.shape) # b 512 7 20

        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x

    def resizing(self, img, mode='bilinear', height=128, width=128):
        '''
        img : (b, c, H, W) dtype (float32)
        out : (b, c, h', w')
        '''
        b, c, h, w = img.shape
        # Resizing
        h = torch.linspace(-1,1,height, device=img.device)
        w = torch.linspace(-1,1,width, device=img.device)
        meshy, meshx = torch.meshgrid((h,w))
        grid = torch.stack((meshx, meshy), 2) # (128, 128, 2(x,y))
        grid = grid.unsqueeze(0) # (1, 128, 128, 2)
        grid = grid.repeat(b, 1,1,1)

        out = F.grid_sample(img, grid, mode=mode, align_corners=True)
        return out

class VGG16_fc1_concat(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            # torch.nn.init.kaiming_uniform_(module.weight)
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            # torch.nn.init.kaiming_uniform_(module.weight)
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_fc1_concat, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(184* 125*335, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        f1 = self.feature[0](x)
        f2 = self.feature[0:2](x)
        f3 = self.feature[0:3](x)
        f4 = self.feature[0:4](x)
        f5 = self.feature[0:5](x)

        b, c, h, w = f1.shape
        up_f2 = self.resizing(f2, height=h, width=w)
        up_f3 = self.resizing(f3, height=h, width=w)
        up_f4 = self.resizing(f4, height=h, width=w)
        up_f5 = self.resizing(f5, height=h, width=w)
        concat_f = torch.cat([f1, up_f2, up_f3, up_f4, up_f5], dim=1)
        x = concat_f

        # x = self.feature(x)
        # print(x.shape) # b 512 7 20

        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x

    def resizing(self, img, mode='bilinear', height=128, width=128):
        '''
        img : (b, c, H, W) dtype (float32)
        out : (b, c, h', w')
        '''
        b, c, h, w = img.shape
        # Resizing
        h = torch.linspace(-1,1,height, device=img.device)
        w = torch.linspace(-1,1,width, device=img.device)
        meshy, meshx = torch.meshgrid((h,w))
        grid = torch.stack((meshx, meshy), 2) # (128, 128, 2(x,y))
        grid = grid.unsqueeze(0) # (1, 128, 128, 2)
        grid = grid.repeat(b, 1,1,1)

        out = F.grid_sample(img, grid, mode=mode, align_corners=True)
        return out
    
class VGG16_fc2_concat(nn.Module):
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            # torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
            # torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __init__(self, base_dim, num_classes=6):
        super(VGG16_fc2_concat, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(1,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            conv_3_block(8*base_dim,8*base_dim), #512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(184* 125*335, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        f1 = self.feature[0](x)
        f2 = self.feature[0:2](x)
        f3 = self.feature[0:3](x)
        f4 = self.feature[0:4](x)
        f5 = self.feature[0:5](x)

        b, c, h, w = f1.shape
        up_f2 = self.resizing(f2, height=h, width=w)
        up_f3 = self.resizing(f3, height=h, width=w)
        up_f4 = self.resizing(f4, height=h, width=w)
        up_f5 = self.resizing(f5, height=h, width=w)
        concat_f = torch.cat([f1, up_f2, up_f3, up_f4, up_f5], dim=1)
        x = concat_f

        # x = self.feature(x)
        # print(x.shape) # b 512 7 20

        x = x.view(x.size(0), -1)
        # print(x.shape) # b 71680
        x = self.fc_layer(x)
        return x

    def resizing(self, img, mode='bilinear', height=128, width=128):
        '''
        img : (b, c, H, W) dtype (float32)
        out : (b, c, h', w')
        '''
        b, c, h, w = img.shape
        # Resizing
        h = torch.linspace(-1,1,height, device=img.device)
        w = torch.linspace(-1,1,width, device=img.device)
        meshy, meshx = torch.meshgrid((h,w))
        grid = torch.stack((meshx, meshy), 2) # (128, 128, 2(x,y))
        grid = grid.unsqueeze(0) # (1, 128, 128, 2)
        grid = grid.repeat(b, 1,1,1)

        out = F.grid_sample(img, grid, mode=mode, align_corners=True)
        return out