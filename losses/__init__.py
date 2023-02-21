import torch.nn as nn
from importlib import import_module
from utility import Module_CharbonnierLoss


class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()
        self.regularize = []
        self.register_buffer('last_loss_sum', None, False)
        self.last_loss_sum = None

        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')  # type: (str, str)
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'Charb':
                loss_function = Module_CharbonnierLoss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('losses.vgg')
                loss_function = getattr(module, 'VGG')()
            elif loss_type.find('GAN') >= 0:
                module = import_module('losses.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
            elif loss_type in ['g_Spatial', 'g_Occlusion', 'Lw', 'Ls']\
                    or loss_type.startswith("g_"):
                self.regularize.append({
                    'type': loss_type,
                    'weight': float(weight),
                    'last_val': None}
                )
                continue

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function,
                'last_val': None}
            )

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        for r in self.regularize:
            print('{:.3f} * {}'.format(r['weight'], r['type']))

        self.loss_module.to('cuda')

    def forward(self, output, gt, input_frames):
        losses = []
        for l in self.loss:
            if l['function'] is not None:
                if l['type'] == 'T_WGAN_GP' or l['type'] == 'FI_GAN':
                    loss = l['function'](output['frame1'], gt, input_frames)
                    effective_loss = l['weight'] * loss
                    losses.append(effective_loss)
                else:
                    loss = l['function'](output['frame1'], gt)
                    effective_loss = l['weight'] * loss
                    losses.append(effective_loss)
                l['last_val'] = loss.detach().clone()

        for r in self.regularize:
            loss = output[r['type']]
            effective_loss = r['weight'] * loss
            losses.append(effective_loss)
            r['last_val'] = loss.detach().clone()
        loss_sum = sum(losses)
        self.last_loss_sum = loss_sum.detach().clone()

        return loss_sum

    def get_last_stat(self) -> str:
        if self.last_loss_sum is None:
            return "No loss stat"

        head = f"Loss({self.last_loss_sum.item():.3})="

        loss_items = []
        for l in self.loss:
            loss_items.append(f"{l['weight']:.2}*{l['type']}({l['last_val'].item():.3})")

        for r in self.regularize:
            loss_items.append(f"{r['weight']:.2}*{r['type']}({r['last_val'].item():.3})")

        return head + '+'.join(loss_items)