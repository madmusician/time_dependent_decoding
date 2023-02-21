import cupy
import numpy
from tqdm import tqdm

import TestModule
import datareader
from datareader import preset_dataset_factory
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
import torch
from TestModule import preset_test_factory
import losses
import datetime
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='AdaCoF-Pytorch')

# parameters
# Model Selection
parser.add_argument('--model', type=str, default='adacofnet')

# Hardware Setting
parser.add_argument('--gpu_id', type=int)

# Directory Setting
# parser.add_argument('--train_dataset_name', type=str, default='gopro_flavr_train', choices=datareader.preset_dataset_factory.keys())
parser.add_argument('--out_dir', type=str, default='./record/baseline_atvfi')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--test_dataset_name', type=str, default='vimeo90ktri', choices=TestModule.preset_test_factory.keys())

# Learning Options
parser.add_argument('--epochs', type=int, default=15, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--loss', type=str, default='1*Charb+0.01*g_Spatial+0.005*g_Occlusion', help='loss function configuration')
# parser.add_argument('--patch_size', type=int, default=256, help='Patch size')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=6, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAMax', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'), help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Options for AdaCoF
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

# parser.add_argument('--train_sample_t', type=int)  # has been always 1
parser.add_argument('--max_inter_frames', type=int, default=8, help='the maximum window width')
parser.add_argument('--gopro_use_augment', action='store_true', help='use augmentation on GoPro dataset')
parser.add_argument('--skip_initial_test', action='store_true', help='skip the test before training to save time')

transform = transforms.Compose([transforms.ToTensor()])


class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Subnet_offset(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64 + 1, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        def Subnet_weight(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64 + 1, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        def Subnet_occlusion():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64 + 1, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                torch.nn.Sigmoid()
            )

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512 + 1, 512)
        self.moduleUpsample5 = Upsample(512)

        self.moduleDeconv4 = Basic(512 + 1, 256)
        self.moduleUpsample4 = Upsample(256)

        self.moduleDeconv3 = Basic(256 + 1, 128)
        self.moduleUpsample3 = Upsample(128)

        self.moduleDeconv2 = Basic(128 + 1, 64)
        self.moduleUpsample2 = Upsample(64)

        self.moduleWeight1 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleWeight2 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha2 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta2 = Subnet_offset(self.kernel_size ** 2)
        self.moduleOcclusion = Subnet_occlusion()

    def forward(self, rfield0, rfield2, ts=None):
        assert isinstance(ts, (torch.Tensor, type(None)))

        tensorJoin = torch.cat([rfield0, rfield2], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        kernels = []

        if ts is None:
            ts = tensorPool5.new_full((tensorPool5.shape[0], 1), 0.5)
        if ts.ndim == 1:
            ts = ts.reshape((-1, 1))
        assert ts.ndim == 2 and ts.shape[0] == tensorPool5.shape[0]  # in shape (bsz, t_sample)

        for t in torch.unbind(ts, 1):
            # t in shape (bsz,)
            t = t.view((-1, 1, 1, 1))

            def combine_t(tensor: torch.Tensor) -> torch.Tensor:
                bsz, _, fh, fw = tensor.shape
                t_feat = torch.broadcast_to(t, (bsz, 1, fh, fw))
                return torch.cat([tensor, t_feat], dim=1)

            tensorPool5t = combine_t(tensorPool5)
            tensorDeconv5 = self.moduleDeconv5(tensorPool5t)
            tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

            tensorCombine = tensorUpsample5 + tensorConv5

            tensorCombine = combine_t(tensorCombine)
            tensorDeconv4 = self.moduleDeconv4(tensorCombine)
            tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

            tensorCombine = tensorUpsample4 + tensorConv4

            tensorCombine = combine_t(tensorCombine)
            tensorDeconv3 = self.moduleDeconv3(tensorCombine)
            tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

            tensorCombine = tensorUpsample3 + tensorConv3

            tensorCombine = combine_t(tensorCombine)
            tensorDeconv2 = self.moduleDeconv2(tensorCombine)
            tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

            tensorCombine = tensorUpsample2 + tensorConv2

            tensorCombine = combine_t(tensorCombine)

            Weight1 = self.moduleWeight1(tensorCombine)
            Alpha1 = self.moduleAlpha1(tensorCombine)
            Beta1 = self.moduleBeta1(tensorCombine)
            Weight2 = self.moduleWeight2(tensorCombine)
            Alpha2 = self.moduleAlpha2(tensorCombine)
            Beta2 = self.moduleBeta2(tensorCombine)
            Occlusion = self.moduleOcclusion(tensorCombine)

            kernels.append((Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion))

        return kernels


class AdaCoFNet(torch.nn.Module):
    def __init__(self, args):
        from cupy_module import adacof
        super(AdaCoFNet, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        self.dilation = args.dilation

        self.get_kernel = KernelEstimation(self.kernel_size)

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

    def forward(self, frame0, frame2, ts=None,):
        from utility import CharbonnierFunc, moduleNormalize

        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, (0, 0, 0, pad_h), mode='reflect')
            frame2 = F.pad(frame2, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, (0, pad_w, 0, 0), mode='reflect')
            frame2 = F.pad(frame2, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True

        out_frames = []
        g_Spatial = 0.
        g_Occlusion = 0.

        kernels = self.get_kernel(moduleNormalize(frame0), moduleNormalize(frame2), ts)
        for Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion in kernels:
            tensorAdaCoF1 = self.moduleAdaCoF(self.modulePad(frame0), Weight1, Alpha1, Beta1, self.dilation)
            tensorAdaCoF2 = self.moduleAdaCoF(self.modulePad(frame2), Weight2, Alpha2, Beta2, self.dilation)

            frame1 = Occlusion * tensorAdaCoF1 + (1 - Occlusion) * tensorAdaCoF2
            if h_padded:
                frame1 = frame1[:, :, 0:h0, :]
            if w_padded:
                frame1 = frame1[:, :, :, 0:w0]
            out_frames.append(frame1)

            if self.training:
                # Smoothness Terms
                m_Alpha1 = torch.mean(Weight1 * Alpha1, dim=1, keepdim=True)
                m_Alpha2 = torch.mean(Weight2 * Alpha2, dim=1, keepdim=True)
                m_Beta1 = torch.mean(Weight1 * Beta1, dim=1, keepdim=True)
                m_Beta2 = torch.mean(Weight2 * Beta2, dim=1, keepdim=True)

                g_Alpha1 = CharbonnierFunc(m_Alpha1[:, :, :, :-1] - m_Alpha1[:, :, :, 1:]) + CharbonnierFunc(m_Alpha1[:, :, :-1, :] - m_Alpha1[:, :, 1:, :])
                g_Beta1 = CharbonnierFunc(m_Beta1[:, :, :, :-1] - m_Beta1[:, :, :, 1:]) + CharbonnierFunc(m_Beta1[:, :, :-1, :] - m_Beta1[:, :, 1:, :])
                g_Alpha2 = CharbonnierFunc(m_Alpha2[:, :, :, :-1] - m_Alpha2[:, :, :, 1:]) + CharbonnierFunc(m_Alpha2[:, :, :-1, :] - m_Alpha2[:, :, 1:, :])
                g_Beta2 = CharbonnierFunc(m_Beta2[:, :, :, :-1] - m_Beta2[:, :, :, 1:]) + CharbonnierFunc(m_Beta2[:, :, :-1, :] - m_Beta2[:, :, 1:, :])
                g_Occlusion = g_Occlusion + CharbonnierFunc(Occlusion[:, :, :, :-1] - Occlusion[:, :, :, 1:]) + CharbonnierFunc(Occlusion[:, :, :-1, :] - Occlusion[:, :, 1:, :])

                g_Spatial = g_Spatial + g_Alpha1 + g_Beta1 + g_Alpha2 + g_Beta2

        if self.training:
            return {'frame1': out_frames, 'g_Spatial': g_Spatial, 'g_Occlusion': g_Occlusion}
        else:
            return out_frames[0]


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.model = AdaCoFNet(args)
        self.model.cuda()

    def forward(self, *frames, **extra):
        return self.model(*frames, **extra)  # XXX: make frames explicit

    def load(self, state_dict):
        load_res = self.model.load_state_dict(state_dict, strict=False)
        print(f"loaded model, resulting {load_res}")

    def get_state_dict(self):
        return self.model.state_dict()

    def get_kernel(self, frame0, frame1):
        return self.model.get_kernel(frame0, frame1)


class GoProFlatMapGTWithTime(torch.utils.data.Dataset):
    """
    A GoPro dataset adapter, which "flat_map" each item `(frames, gt_frames)` of [datareader.GoPro],
    into a series of (frames, gt_frames[0], ts[0]), (frames, gt_frames[1], ts[1]), ...
    """
    def __init__(self, n_inter_frames: int, *, use_augment: bool = False):
        from datareader import GoPro
        self.dataset = GoPro('./db/gopro', 'train', interFrames=n_inter_frames, n_inputs=2, use_augment=use_augment)
        self.n_inter_frames = n_inter_frames
        self.all_ts = numpy.arange(1, n_inter_frames + 1, dtype=numpy.float32) / (n_inter_frames + 1)

        # reshape so that `__getitem__` return t of ndarray with shape (1,), then get collated into (bsz, 1)
        self.all_ts = self.all_ts.reshape((-1, 1))

    def __getitem__(self, idx):
        frames, gt_frames = self.dataset[idx // self.n_inter_frames]
        assert len(gt_frames) == self.n_inter_frames

        it = idx % self.n_inter_frames
        gt_frame = gt_frames[it]
        t = self.all_ts[it]

        return frames, gt_frame, t

    def __len__(self):
        return len(self.dataset) * self.n_inter_frames


class VariantWindowGoPro(torch.utils.data.ConcatDataset):
    def __init__(self, max_num_inter_frames: int, *, use_augment: bool = False):
        # Concat Gopro(interFrames=1), GoPro(interFrames=2), ...
        datasets = [GoProFlatMapGTWithTime(n_inter_frame, use_augment=use_augment)
                    for n_inter_frame in range(1, max_num_inter_frames + 1)]

        super(VariantWindowGoPro, self).__init__(datasets)


class Trainer:
    def __init__(self, args, train_loader, test_loader, my_model, my_loss, start_epoch=0):
        import utility
        import os

        self.args = args
        self.train_loader = train_loader
        self.max_step = self.train_loader.__len__()
        self.test_loader = test_loader
        self.model = my_model
        self.loss = my_loss
        self.current_epoch = start_epoch

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        self.result_dir = args.out_dir + '/result'
        self.ckpt_dir = args.out_dir + '/checkpoint'

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.logfile = open(args.out_dir + '/log.txt', 'w')

        # Initial Test
        if not args.skip_initial_test:
            self.model.eval()
            self.test_loader.Test(self.model, self.result_dir, self.current_epoch, self.logfile, str(self.current_epoch).zfill(3) + '.png')

    def train(self):
        # Train
        from utility import to_variable

        self.model.train()
        for batch_idx, (input_frames, gt_frame, ts) in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()

            frame0, frame2 = map(to_variable, input_frames)
            # gt_frames = [to_variable(frame) for frame in gt_frames]
            gt_frame = to_variable(gt_frame)
            ts = to_variable(ts)

            output = self.model(frame0, frame2, ts=ts)

            output['frame1'] = torch.cat(output['frame1'])
            # gt_frame = torch.cat(gt_frames)

            loss = self.loss(output, gt_frame, [frame0, frame2])

            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                tqdm.write('{:<13s}{:<14s}{:<6s}{:<16s}{:s}'.format('Train Epoch: ', '[' + str(self.current_epoch) + '/' + str(self.args.epochs) + ']', 'Step: ', '[' + str(batch_idx) + '/' + str(self.max_step) + ']', self.loss.get_last_stat()))
        self.current_epoch += 1
        self.scheduler.step()

    def test(self):
        # Test
        torch.save({'epoch': self.current_epoch, 'state_dict': self.model.get_state_dict()}, self.ckpt_dir + '/model_epoch' + str(self.current_epoch).zfill(3) + '.pth')
        self.model.eval()
        self.test_loader.Test(self.model, self.result_dir, self.current_epoch, self.logfile, str(self.current_epoch).zfill(3) + '.png')
        self.logfile.write('\n')
        self.logfile.flush()

    def terminate(self):
        return self.current_epoch >= self.args.epochs

    def close(self):
        self.logfile.close()


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    # dataset = preset_dataset_factory[args.train_dataset_name](use_augment=args.gopro_use_augment)
    dataset = VariantWindowGoPro(args.max_inter_frames, use_augment=args.gopro_use_augment)
    # TestDB = Middlebury_other(args.test_input, args.gt)
    TestDB = preset_test_factory[args.test_dataset_name]()
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = Model(args)
    loss = losses.Loss(args)

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load, map_location=f"cuda:{args.gpu_id}")
        model.load(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    my_trainer = Trainer(args, train_loader, TestDB, model, loss, start_epoch)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(args.out_dir + '/config.txt', 'a') as f:
        f.write(now + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')

    while not my_trainer.terminate():
        my_trainer.train()
        my_trainer.test()

    my_trainer.close()


if __name__ == "__main__":
    main()
