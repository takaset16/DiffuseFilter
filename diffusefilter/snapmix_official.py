"""SnapMix (Huang et al., AAAI 2021), re-implemented from the official code
(github.com/Shaoli-Huang/SnapMix: utils/mixmethod.py get_spm / snapmix, utils/custom_ops.py rand_bbox)
for torchvision ResNets.

- The semantic percentage map (SPM) is the class activation map of the model being trained: the ReLU'd last
  feature map is multiplied by the fc weights and bias (1x1 convolution), the map of the target class is
  upsampled bilinearly to the input size, shifted to a minimum of zero and normalized to sum one.
  Computed without gradients with the model in train mode, as in the official train loop.
- The target box (lam) and the source box (lam1) are drawn from separate Beta(beta, beta); the source crop is
  resized bilinearly to the target box.
- Label weights: lam_a = 1 - SPM mass inside the target box, lam_b = partner's SPM mass inside the source box;
  for same-label pairs both are summed; NaN falls back to the area ratio.
- The official loss mean(lam_a * CE(y_a) + lam_b * CE(y_b)) equals the soft-label cross entropy with
  t = lam_a * onehot(y_a) + lam_b * onehot(y_b), because cross entropy is linear in the target.
- Defaults follow the official README (beta = 5, prob = 1). Images stay normalized, as in the official code.
With the same random numbers, this module reproduces the official functions exactly (checked for ResNet-18/50).
"""
import numpy as np
import torch
import torch.nn.functional as F


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


def _features_and_classifier(model, x):
    """Last convolutional feature map (before pooling) and the fc layer of a torchvision ResNet."""
    m = _unwrap(model)
    if not all(hasattr(m, k) for k in ('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'fc')):
        raise ValueError('SnapMix supports torchvision ResNets only')
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
    return x, m.fc


def get_spm(images, target, model):
    h, w = images.shape[-2:]
    bs = images.size(0)
    with torch.no_grad():
        fms, clsw = _features_and_classifier(model, images)
        fms = F.relu(fms.float())
        weight = clsw.weight.data.float()
        bias = clsw.bias.data.float()
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)
        out = F.conv2d(fms, weight, bias=bias)
        outmaps = out[torch.arange(bs, device=out.device), target]
        outmaps = outmaps.view(bs, 1, outmaps.size(1), outmaps.size(2))
        outmaps = F.interpolate(outmaps, (h, w), mode='bilinear', align_corners=False)
        outmaps = outmaps.view(bs, h, w)
        outmaps = outmaps - outmaps.view(bs, -1).min(dim=1)[0].view(bs, 1, 1)
        outmaps = outmaps / outmaps.view(bs, -1).sum(dim=1).view(bs, 1, 1)
    return outmaps


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def snapmix(images, target, model, num_classes, beta=5.0, prob=1.0):
    """images: normalized (B, C, H, W); target: (B,) class indices.
    Returns the mixed images and soft labels (B, num_classes)."""
    bs = images.size(0)
    dev = images.device
    lam_a = torch.ones(bs, device=dev)
    lam_b = 1 - lam_a
    target_b = target.clone()
    images = images.clone()

    r = np.random.rand(1)
    if r < prob:
        wfmaps = get_spm(images, target, model)
        lam = np.random.beta(beta, beta)
        lam1 = np.random.beta(beta, beta)
        rand_index = torch.randperm(bs, device=dev)
        wfmaps_b = wfmaps[rand_index, :, :]
        target_b = target[rand_index]

        same_label = target == target_b
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(images.size(), lam1)

        area = (bby2 - bby1) * (bbx2 - bbx1)
        area1 = (bby2_1 - bby1_1) * (bbx2_1 - bbx1_1)

        if area1 > 0 and area > 0:
            ncont = images[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont.float(), size=(int(bbx2 - bbx1), int(bby2 - bby1)),
                                  mode='bilinear', align_corners=True).to(images.dtype)
            images[:, :, bbx1:bbx2, bby1:bby2] = ncont
            lam_a = 1 - wfmaps[:, bbx1:bbx2, bby1:bby2].sum(2).sum(1) / (wfmaps.sum(2).sum(1) + 1e-8)
            lam_b = wfmaps_b[:, bbx1_1:bbx2_1, bby1_1:bby2_1].sum(2).sum(1) / (wfmaps_b.sum(2).sum(1) + 1e-8)
            tmp = lam_a.clone()
            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            lam_a[torch.isnan(lam_a)] = float(lam)
            lam_b[torch.isnan(lam_b)] = 1 - float(lam)

    eye = torch.eye(num_classes, device=dev, dtype=torch.float32)
    soft = lam_a.view(bs, 1).float() * eye[target] + lam_b.view(bs, 1).float() * eye[target_b]
    return images, soft
