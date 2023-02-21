import argparse
import torch
import os
import TestModule

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--checkpoint', type=str, default='./record/baseline_atvfi/checkpoint/model_epoch015.pth')
parser.add_argument('--config', type=str, default='./record/baseline_atvfi/config.txt')
parser.add_argument('--out_dir', type=str, default='./output_adacof_test')

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    if args.config is not None:
        config_file = open(args.config, 'r')
        while True:
            line = config_file.readline()
            if not line:
                break
            if line.find(':') == 0:
                continue
            else:
                tmp_list = line.split(': ')
                if tmp_list[0] == 'kernel_size':
                    args.kernel_size = int(tmp_list[1])
                if tmp_list[0] == 'dilation':
                    args.dilation = int(tmp_list[1])
        config_file.close()

    # model = models.Model(args)
    from run_train import Model as Model
    model = Model(args)

    print('Loading the model...')

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load(checkpoint['state_dict'])
    current_epoch = checkpoint['epoch']

    print('Test: Middlebury_eval')
    test_dir = args.out_dir + '/middlebury_eval'
    test_db = TestModule.Middlebury_eval('./test_input/middlebury_eval')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db.Test(model, test_dir)

    print('Test: Middlebury_others')
    test_dir = args.out_dir + '/middlebury_others'
    test_db = TestModule.Middlebury_other('./test_input/middlebury_others/input', './test_input/middlebury_others/gt')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db.Test(model, test_dir, current_epoch, output_name='frame10i11.png')

    print('Test: UCF101')
    test_dir = args.out_dir + '/ucf101'
    test_db = TestModule.preset_test_factory['ucf101qvi4as2']()
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db.Test(model, test_dir, current_epoch, output_name='frame1.png')

    print('Test: Vimeo90k Triplet')
    test_db = TestModule.vimeo90ktri()
    test_db.Test(model, test_dir, current_epoch, output_name='frame1.png')


if __name__ == "__main__":
    main()
