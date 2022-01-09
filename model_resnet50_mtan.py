import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler

from create_dataset import *
from utils import *

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa, pcgrad')
parser.add_argument('--dataroot', default='/data/dataset/nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--apply_weightdeacay', action='store_true', help='toggle to apply weight decay 1e-5 on learning rate')
parser.add_argument('--apply_pcgrad', action='store_true', help='toggle to apply gradient surgery')

opt = parser.parse_args()


import torch
import resnet

from resnet_dilated import ResnetDilated
#from aspp import DeepLabHead
from resnet import Bottleneck, conv1x1


class MTANDeepLabv3(nn.Module):
    def __init__(self):
        super(MTANDeepLabv3, self).__init__()
        backbone = ResnetDilated(resnet.__dict__['resnet18'](pretrained=True))
        ch = [64,128,256,512]
        #ch = [256, 512, 1024, 2048]

        self.class_nb = 13
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        self.tasks = ['1', '2', '3', '4', '5','6','7','8','9','10']
        #self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}

        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)

        # We will apply the attention over the last bottleneck layer in the ResNet.
        self.shared_layer1_b = backbone.layer1[:-1]
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0], ch[0] // 4, ch[0]) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)

        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classfier = nn.ModuleList([nn.Linear(512, 10) for _ in self.tasks])
         # or 20

        # Define task-specific decoders using ASPP modules
        #self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])

    # def forward(self, x, out_size):
    def forward(self, x, task_id):
        img_size  = x.size()[-2:]
        # Shared convolution
        x = self.shared_conv(x)

        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)

        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]

        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]

        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]

        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]

        for i, t in enumerate(self.tasks):
            if t == task_id:
                z = torch.flatten(a_4[i], 1)
                out = self.classfier[i](z)
                return out

    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())

    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return Bottleneck(in_channel, out_channel, downsample=downsample)

'''
        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            # out[i] = F.interpolate(self.decoders[i](a_4[i]), size=out_size, mode='bilinear', align_corners=True)
            out[i] = F.interpolate(self.decoders[i](a_4[i]), img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return [out[0], out[1], out[2]], self.logsigma
'''




# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Resnet_MTAN = MTANDeepLabv3().to(device)

from pcgrad import PCGrad

if opt.apply_pcgrad:
    optimizer = optim.Adam(Resnet_MTAN.parameters(), lr=1e-4)
    weight_optimizer = PCGrad(optimizer)
    print('Applying gradient surgery on optimizer.')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
else:
    if opt.apply_weightdeacay:
        optimizer = optim.Adam(Resnet_MTAN.parameters(), lr=1e-4, weight_decay=1e-5)
        print('Applying weight decay 1e-5 on learning rate.')
    else:
        optimizer = optim.Adam(Resnet_MTAN.parameters(), lr=1e-4)
        print('Not applying weight decay on learning rate.')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(Resnet_MTAN),
                                                         count_parameters(Resnet_MTAN) / 24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

# define dataset

dataset_path = opt.dataroot
if opt.apply_augmentation:
    nyuv2_train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
    print('Applying data augmentation on NYUv2.')
else:
    nyuv2_train_set = NYUv2(root=dataset_path, train=True)
    print('Standard training strategy without data augmentation.')

nyuv2_test_set = NYUv2(root=dataset_path, train=False)

batch_size = 8
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False)

# Train and evaluate multi-task network
if opt.apply_pcgrad:
    multi_task_trainer_pcgrad(nyuv2_train_loader,
                       nyuv2_test_loader,
                       Resnet_MTAN,
                       device,
                       weight_optimizer,
                       scheduler,
                       opt,
                       200)
else:
    multi_task_trainer(nyuv2_train_loader,
                       nyuv2_test_loader,
                       Resnet_MTAN,
                       device,
                       optimizer,
                       scheduler,
                       opt,
                       200)

