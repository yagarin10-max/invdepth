import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, output):
        h_tv = ((output[:,:,1:,:] - output[:,:,:-1,:]).pow(2)).sum()
        w_tv = ((output[:,:,:,1:] - output[:,:,:,:-1]).pow(2)).sum()
        return h_tv + w_tv

class TGVLoss(nn.Module):
    def __init__(self, alpha0=1.0, alpha1=2.0):
        super(TGVLoss, self).__init__()
        self.alpha0 = alpha0  # 1次微分の重み
        self.alpha1 = alpha1  # 2次微分の重み

    def forward(self, x):
        # 1次微分
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]

        # 2次微分
        dxx = dx[..., :, 1:] - dx[..., :, :-1]
        dyy = dy[..., 1:, :] - dy[..., :-1, :]
        dxy = dx[..., 1:, :] - dx[..., :-1, :]

        # TGVの計算
        v1 = self.alpha0 * (torch.sum(dx.pow(2)) + torch.sum(dy.pow(2)))
        v2 = self.alpha1 * (torch.sum(dxx.pow(2)) + torch.sum(dyy.pow(2)) + torch.sum(dxy.pow(2)))

        return v1 + v2

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, output, target):
        output_h_edge = torch.roll(output, shifts=1, dims=3) - output
        output_v_edge = torch.roll(output, shifts=1, dims=2) - output
        target_h_edge = torch.roll(target, shifts=1, dims=3) - target
        target_v_edge = torch.roll(target, shifts=1, dims=2) - target

        return torch.abs(output_h_edge - target_h_edge).sum() + torch.abs(output_v_edge - target_v_edge).sum()


class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()
    
    def forward(self, output, target):
        # Compute the Fourier Transform of the output and target images
        output_fft = torch.fft.fft2(output)
        target_fft = torch.fft.fft2(target)
        # Convert the complex tensors to real tensors
        output_fft = torch.view_as_real(output_fft)
        target_fft = torch.view_as_real(target_fft)
        # Compute the loss in the frequency domain
        loss = F.mse_loss(output_fft, target_fft)
        return loss


class VGG19Loss(nn.Module):
    def __init__(self, device="cuda:0"):
        super(VGG19Loss, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval()
        self.vgg = self.vgg.to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        input_vgg = self.vgg(input)
        target_vgg = self.vgg(target)

        return self.mse_loss(input_vgg, target_vgg)


if __name__ == "__main__":
    x = torch.empty((5, 3, 17, 17))
    y = torch.empty((5, 3, 17, 17))
    vggloss = VGG19Loss()
    loss = vggloss(x, y)
