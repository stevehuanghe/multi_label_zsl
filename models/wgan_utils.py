import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


def calc_gradient_penalty(netD, real_data, fake_data, lambda_gp=10.0):
    batch_size = real_data.size(0)
    device = real_data.device
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = (alpha * real_data + ((1.0 - alpha) * fake_data)).requires_grad_(True)

    disc_interpolates = netD(interpolates)

    outputs = torch.ones(disc_interpolates.size()).to(device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=outputs,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


def calc_gradient_penalty_cd(netD, real_data, fake_data, labels, lambda_gp=10.0):
    batch_size = real_data.size(0)
    device = real_data.device
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = (alpha * real_data + ((1.0 - alpha) * fake_data)).requires_grad_(True)

    disc_interpolates = netD(interpolates, labels)

    outputs = torch.ones(disc_interpolates.size()).to(device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=outputs,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty