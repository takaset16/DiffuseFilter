# Run configurations

`runs.tsv` lists every training run behind the tables of the paper (one row per run: run id, experiment group,
arguments). The arguments use the option names of the internal training script from which `train.py` was derived;
the main correspondences are:

| internal | train.py |
|---|---|
| `--n_data`, `--n_model`, `--loop` | `--dataset`, `--model`, `--seed` |
| `--flag_diffusefilter 1 --mode 102` | `--diffusefilter 1 --composite lfr` |
| `--radius`, `--alpha_generated`, `--alpha_blended`, `--blend_width` | `--radius`, `--alpha`, `--fractal_alpha`, `--blend_width` |
| `--flag_final_mix 1 --alpha_final 0.5` | `--apply_ratio 0.5` |
| `--mode 104` / `105` / `107` / `106` | `--composite fda` / `pixel` / `reinhard` / `colorjitter` |
| `--n_aug 21` / `20` / `11` / `12` / `13` / `15` / `25` | `--baseline cutmix` / `mixup` / `saliency_mix` / `puzzle_mix` / `co_mixup` / `snapmix_area` / `snapmix` |

Groups and tables: g02 main results (Table 3); g03, g12, g15, g18 comparison and combination (Table 4);
g10, g13 DiffuseMix (Table 5); g04, g05, g16 components and filter sources (Table 6); g06, g07, g19 compositing
operator and LFR design (Table 7; `comp_reinhard` used the clipping of v1.0 and is superseded by `comp_reinhard_fixed`);
g08, g09 number of filters and prompts (Table S1); g01 radius x alpha grid on validation data (Figure 6);
g11 ViT-B/16 (Table 3); g14 CIFAR-100 (Table S2; the additional radii 2, 4 and 8 used the g14 settings with `--radius` changed).
The seed-level results are in the supplementary data of the paper.
