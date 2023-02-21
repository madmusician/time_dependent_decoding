import tqdm
from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
import os
from utility import to_variable


class Middlebury_eval:
    def __init__(self, input_dir='./evaluation'):
        self.im_list = ['Backyard', 'Basketball', 'Dumptruck', 'Evergreen', 'Mequon', 'Schefflera', 'Teddy', 'Urban']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/input/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/input/' + item + '/frame11.png')).unsqueeze(0)))

    def Test(self, model, output_dir='./evaluation/output', output_name='frame10i11.png'):
        model.eval()
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))


class Middlebury_other:
    def __init__(self, input_dir, gt_dir):
        self.im_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def Test(self, model, output_dir, current_epoch, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', current_epoch) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)


class ucf:
    def __init__(self, input_dir):
        self.im_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame0.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame2.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame1.png')).unsqueeze(0)))

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.get_epoch()) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)


class ucf_qvi:
    def __init__(self, input_dir, as_2_frame=False):
        self.im_list = [str(i) for i in range(100)]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.as_2_frame = as_2_frame

        self.input0_list = []
        self.input1_list = []
        self.input2_list = []
        self.input3_list = []
        self.gt_list = []
        for item in tqdm.tqdm(self.im_list, 'Loading UCF'):
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame0.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame1.png')).unsqueeze(0)))
            self.input2_list.append(
                to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame2.png')).unsqueeze(0)))
            self.input3_list.append(
                to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame3.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/framet.png')).unsqueeze(0)))

    def Test(self, model, output_dir, current_epoch, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', current_epoch) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            if self.as_2_frame:
                frame_out = model(self.input1_list[idx], self.input2_list[idx])
            else:
                frame_out = model(self.input0_list[idx], self.input1_list[idx], self.input2_list[idx], self.input3_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)


class gopro:
    def __init__(self, inter_frames: int = 7, take_input: int = 4, take_output: int = 7):
        self.take_input = take_input  # How many frames does the model inputs? 2 or 4
        self.take_output = take_output  # How many frames does the model outputs? 1 or 7
        self.transform = transforms.Compose([transforms.ToTensor()])

        import datareader
        test_dataset_name = { 7: 'gopro_flavr_test', 1: 'gopro_flavr_test_group7' }[inter_frames]
        dataset = datareader.preset_dataset_factory[test_dataset_name]()  # type: datareader.DBreader_Vimeo90k
        self.im_list_len = len(dataset)

        from torch.utils.data import DataLoader
        self.loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4)

    def Test(self, model, output_dir, current_epoch, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        import numpy
        av_psnr_per_t = numpy.zeros((self.take_output,), dtype=numpy.float64)
        from utility import to_variable

        if logfile is not None:
            logfile.write('{:<7s}{:<3d} testing GoPro {:d}->{:d}'.format('Epoch: ', current_epoch, self.take_input, self.take_output) + '\n')

        kwargs = {}
        if self.model_needs_ts(model):
            ts = [i / 8 for i in range(1, 8)]
            ts = self.pick_from_center(ts, self.take_output)
            ts = [ts]
            ts = torch.tensor(ts, dtype=torch.float32)
            ts = to_variable(ts)
            kwargs['ts'] = ts  # [[0.125, 0.25, ..., 0.875]]

        for idx, (images, gt_images) in enumerate(tqdm.tqdm(self.loader)):
            images = self.pick_from_center(images, self.take_input)
            gt_images = self.pick_from_center(gt_images, self.take_output)

            images = [to_variable(image) for image in images]
            gt_images = [to_variable(image) for image in gt_images]

            with torch.no_grad():
                frame_out = model(*images, **kwargs)

                if self.take_output > 1:
                    assert isinstance(frame_out, list) and len(frame_out) == len(gt_images)
                    for i, (frame_out_per_t, gt_image_per_t) in enumerate(zip(frame_out, gt_images)):
                        av_psnr_per_t[i] += -10 * log10(torch.mean((gt_image_per_t - frame_out_per_t) * (gt_image_per_t - frame_out_per_t)).item())

                if isinstance(frame_out, list):
                    frame_out = torch.cat(frame_out)

                gt = torch.cat(gt_images)

                psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr

        av_psnr /= self.im_list_len
        av_psnr_per_t /= self.im_list_len
        msg = '{:<15s}{:<20.16f}\nAverage per t:{}'.format('Average: ', av_psnr, av_psnr_per_t)
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)

    @staticmethod
    def pick_from_center(ls: list, n: int):
        """ Pick n centermost elements from list """
        assert len(ls) % 2 == n % 2 and n > 0
        center = len(ls) // 2
        radius = n // 2
        return ls[center - radius: center + n - radius]

    @staticmethod
    def model_needs_ts(m: torch.nn.Module):
        import inspect
        return 'ts' in inspect.signature(m.model.forward).parameters


class gopro_iterative:
    def __init__(self, inter_frames: int = 7):
        self.transform = transforms.Compose([transforms.ToTensor()])

        import datareader
        assert inter_frames == 7  # others are unimplemented
        test_dataset_name = 'gopro_flavr_test'
        dataset = datareader.preset_dataset_factory[test_dataset_name]()  # type: datareader.DBreader_Vimeo90k
        self.im_list_len = len(dataset)

        from torch.utils.data import DataLoader
        self.loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4)

    def Test(self, model, output_dir, current_epoch, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        import numpy
        av_psnr_per_t = numpy.zeros((7,), dtype=numpy.float64)
        from utility import to_variable

        if logfile is not None:
            logfile.write('{:<7s}{:<3d} testing GoPro Iterative'.format('Epoch: ', current_epoch) + '\n')

        kwargs = {}
        if gopro.model_needs_ts(model):
            import warnings
            warnings.warn("AdaCoF with Time models do not need iterative interpolation")
            # ts = [i / 8 for i in range(1, 8)]
            # ts = self.pick_from_center(ts, self.take_output)
            # ts = [ts]
            # ts = torch.tensor(ts, dtype=torch.float32)
            # ts = to_variable(ts)
            # kwargs['ts'] = ts  # [[0.125, 0.25, ..., 0.875]]

        for idx, (images, gt_images) in enumerate(tqdm.tqdm(self.loader)):
            images = [to_variable(image) for image in images]
            gt_images = [to_variable(image) for image in gt_images]
            assert len(images) == 4 and len(gt_images) == 7

            with torch.no_grad():
                _, i0, i8, _ = images

                i4 = model(i0, i8)

                i2 = model(i0, i4)
                i6 = model(i4, i8)

                i1 = model(i0, i2)
                i3 = model(i2, i4)
                i5 = model(i4, i6)
                i7 = model(i6, i8)

                frame_out = [i1, i2, i3, i4, i5, i6, i7]

                for i, (frame_out_per_t, gt_image_per_t) in enumerate(zip(frame_out, gt_images)):
                    av_psnr_per_t[i] += -10 * log10(
                        torch.mean((gt_image_per_t - frame_out_per_t) * (gt_image_per_t - frame_out_per_t)).item())

                frame_out = torch.cat(frame_out)
                gt = torch.cat(gt_images)

                psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr

        av_psnr /= self.im_list_len
        av_psnr_per_t /= self.im_list_len
        msg = '{:<15s}{:<20.16f}\nAverage per t:{}'.format('Average: ', av_psnr, av_psnr_per_t)
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)

class Middlebury_other4:
    def __init__(self, input_dir, gt_dir, as_2_frame):
        # removed 2 2-frame testing example
        self.im_list = ['Beanbags', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Walking']
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.as_2_frame = as_2_frame

        self.input10_list = []
        self.input11_list = []
        self.input9_list = []
        self.input12_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input10_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input11_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.input9_list.append(
                to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame09.png')).unsqueeze(0)))
            self.input12_list.append(
                to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame12.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def Test(self, model, output_dir, current_epoch, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', current_epoch) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            if self.as_2_frame:
                frame_out = model(self.input10_list[idx], self.input11_list[idx])
            else:
                frame_out = model(self.input9_list[idx], self.input10_list[idx], self.input11_list[idx], self.input12_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)

class vimeo90ktri:
    def __init__(self, ):
        self.transform = transforms.Compose([transforms.ToTensor()])

        import datareader
        dataset = datareader.preset_dataset_factory['vimeo_triplet_test']()  # type: datareader.DBreader_Vimeo90k
        self.im_list = [im_path[-len('00001/0001'):] for im_path in dataset.triplet_list]
        assert len(self.im_list) == len(dataset)
        from torch.utils.data import DataLoader
        self.loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4)

    def Test(self, model, output_dir, current_epoch, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        from utility import to_variable
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', current_epoch) + '\n')
        for idx, images in enumerate(tqdm.tqdm(self.loader)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            images = [to_variable(image) for image in images]

            img0, gt, img2 = images
            frame_out = model(img0, img2)

            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            if idx % 100 == 0:
                imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
                msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
                tqdm.tqdm.write(msg, end='')
                if logfile is not None:
                    logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)


preset_test_factory = {
    "ucf101qvi": lambda: ucf_qvi('./test_input/ucf101_extracted'),
    "ucf101qvi4as2": lambda: ucf_qvi('./test_input/ucf101_extracted', as_2_frame=True),
    "vimeo90ktri": lambda: vimeo90ktri(),
    "gopro_flavr": lambda: gopro(),
    "gopro_flavr_2to7": lambda: gopro(take_input=2),
    "gopro_flavr_4to1": lambda: gopro(take_input=4, take_output=1),
    "gopro_flavr_2to1": lambda: gopro(take_input=2, take_output=1),
    "gopro_flavr_iter2to7": lambda: gopro_iterative(),
}