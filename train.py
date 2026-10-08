import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.amp import GradScaler, autocast
from warmup_scheduler import GradualWarmupScheduler

from diffusefilter.augment import DiffuseFilter
from diffusefilter.baselines import BASELINES, apply_baseline
from diffusefilter.composite import COMPOSITES
from diffusefilter.datasets import DATASETS, NUM_CLASSES, build_datasets
from diffusefilter.filters import load_filters, load_images


def parse_args():
    p = argparse.ArgumentParser(description="Train a classifier with DiffuseFilter or a baseline augmentation.")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--model", default="ResNet18", choices=["ResNet18", "ResNet50", "ViT_B16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=1, help="linear warm-up (3 epochs) before the cosine schedule")
    p.add_argument("--num_workers", type=int, default=4)

    g = p.add_argument_group("DiffuseFilter")
    g.add_argument("--diffusefilter", type=int, default=0)
    g.add_argument("--filter_dir", default="./filters")
    g.add_argument("--fractal_dir", default="./fractals")
    g.add_argument("--composite", default="lfr", choices=COMPOSITES)
    g.add_argument("--alpha", type=float, default=0.4, help="LFR strength")
    g.add_argument("--radius", type=int, default=16, help="LFR mask radius")
    g.add_argument("--mask_shape", default="circle", choices=["circle", "square", "gaussian"])
    g.add_argument("--amp_norm", type=int, default=1, help="amplitude normalization (Eq. 4)")
    g.add_argument("--use_filter", type=int, default=1)
    g.add_argument("--use_concat", type=int, default=1)
    g.add_argument("--use_fractal", type=int, default=1)
    g.add_argument("--blend_width", type=int, default=32)
    g.add_argument("--fractal_alpha", type=float, default=0.05)
    g.add_argument("--apply_ratio", type=float, default=0.5, help="fraction of each batch that is augmented")
    g.add_argument("--filters_per_style", type=int, default=0, help="0: use all")
    g.add_argument("--filter_styles", default="", help="comma-separated substrings; empty: all")
    g.add_argument("--filter_seed", type=int, default=0)

    g = p.add_argument_group("baselines")
    g.add_argument("--baseline", default="none", choices=BASELINES,
                   help="applied after DiffuseFilter when both are given")
    g.add_argument("--da_param1", type=float, default=None)
    g.add_argument("--da_param2", type=float, default=None)
    g.add_argument("--da_param3", type=float, default=None)
    g.add_argument("--da_fixed", type=int, default=0)

    g = p.add_argument_group("DiffuseMix")
    g.add_argument("--diffusemix", default="", choices=["", "only", "combine"])
    g.add_argument("--diffusemix_dir", default="./diffusemix")
    g.add_argument("--dm_max_k", type=int, default=-1)

    g = p.add_argument_group("evaluation")
    g.add_argument("--val_mode", default="none", choices=["none", "holdout"],
                   help="holdout: evaluate on a validation split of the training set")
    g.add_argument("--val_ratio", type=float, default=0.1)
    g.add_argument("--out_dir", default="results")
    g.add_argument("--run_name", default="")
    return p.parse_args()


def build_model(name, num_classes):
    if name == "ResNet18":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "ResNet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def soft_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def main():
    args = parse_args()
    seed = 1001 + args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = NUM_CLASSES[args.dataset]
    train_set, test_set = build_datasets(args.dataset, args.data_root, args.diffusemix or None,
                                         args.diffusemix_dir, args.dm_max_k, args.val_mode, args.val_ratio)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
                     prefetch_factor=4 if args.num_workers > 0 else None)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               drop_last=True, **loader_kw)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **loader_kw)

    augment, filter_info = None, None
    if args.diffusefilter:
        styles = [s.strip() for s in args.filter_styles.split(",") if s.strip()]
        filters, filter_info = load_filters(args.filter_dir, per_style=args.filters_per_style,
                                            styles=styles or None, seed=args.filter_seed)
        fractals = load_images(args.fractal_dir)
        augment = DiffuseFilter(filters.to(device), fractals.to(device), composite=args.composite,
                                alpha=args.alpha, radius=args.radius, mask_shape=args.mask_shape,
                                amp_norm=bool(args.amp_norm), blend_width=args.blend_width,
                                fractal_alpha=args.fractal_alpha, apply_ratio=args.apply_ratio,
                                use_filter=bool(args.use_filter), use_concat=bool(args.use_concat),
                                use_fractal=bool(args.use_fractal))

    model = build_model(args.model, num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.)
    if args.warmup:
        total_epoch = 3
        if args.diffusemix:
            # Shorten the warm-up in proportion to the epoch count (reference: 100 epochs).
            total_epoch = max(1, int(np.ceil(total_epoch * args.epochs / 100)))
        scheduler = GradualWarmupScheduler(optimizer, multiplier=2, total_epoch=total_epoch, after_scheduler=scheduler)
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    accs, losses, epoch_times = [], [], []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if augment is not None:
                images = augment(images)
            images, labels = apply_baseline(args.baseline, images, labels, num_classes,
                                            args.da_param1, args.da_param2, args.da_param3, bool(args.da_fixed))

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                if labels.ndim > 1:
                    loss = soft_cross_entropy(outputs, labels.float())
                else:
                    loss = criterion(outputs, labels.long())
            loss_sum += loss.item() * images.size(0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.time() - t0)
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(images)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
        acc = 100.0 * correct / len(test_set)
        accs.append(acc)
        losses.append(loss_sum / len(train_loader.dataset))
        split = "Val" if args.val_mode == "holdout" else "Test"
        print(f"Epoch [{epoch + 1}/{args.epochs}] train loss {losses[-1]:.4f}  {split} acc {acc:.2f}%  "
              f"time {epoch_times[-1]:.1f}s")

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "eval_split": "val" if args.val_mode == "holdout" else "test",
        "acc_final": accs[-1],
        "acc_last5_mean": float(np.mean(accs[-5:])),
        "acc_best": float(np.max(accs)),
        "acc_curve": accs,
        "train_loss_curve": losses,
        "train_sec_per_epoch": float(np.mean(epoch_times)),
        "total_min": (time.time() - start) / 60.0,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else None,
        "filter_info": filter_info,
        "args": vars(args),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    name = args.run_name or f"{args.dataset}_{args.model}_seed{args.seed}_{int(time.time())}"
    path = os.path.join(args.out_dir, name + ".json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"final acc {accs[-1]:.2f}%  -> {path}")


if __name__ == "__main__":
    main()
