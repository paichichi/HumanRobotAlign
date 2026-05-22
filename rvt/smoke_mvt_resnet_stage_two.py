import argparse

import config as exp_cfg_mod
import rvt.mvt.config as mvt_cfg_mod
from rvt.mvt.mvt import MVT


def apply_exp_overrides_to_mvt_cfg(mvt_cfg, exp_cfg):
    exp_to_mvt = [
        ("depth", "depth"),
        ("attn_dim", "attn_dim"),
        ("ds_rate", "ds_rate"),
        ("adapter", "adapter"),
        ("model", "model"),
        ("output_dim", "output_dim"),
        ("stage_two", "stage_two"),
        ("rot_ver", "rot_ver"),
        ("rot_x_y_aug", "rot_x_y_aug"),
        ("feat_ver", "feat_ver"),
        ("use_point_renderer", "use_point_renderer"),
        ("cvx_up", "cvx_up"),
        ("pretrain", "pretrain_path"),
    ]
    mvt_cfg.defrost()
    for exp_key, mvt_key in exp_to_mvt:
        if hasattr(exp_cfg, exp_key):
            value = getattr(exp_cfg, exp_key)
            if value is not None:
                mvt_cfg[mvt_key] = value
    mvt_cfg.freeze()
    return mvt_cfg


def build(exp_cfg_path, mvt_cfg_path, renderer_device):
    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    exp_cfg.merge_from_file(exp_cfg_path)
    exp_cfg.freeze()

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    mvt_cfg.merge_from_file(mvt_cfg_path)
    mvt_cfg = apply_exp_overrides_to_mvt_cfg(mvt_cfg, exp_cfg)
    print(f"Renderer device: {renderer_device}")
    print(f"MVT config: {mvt_cfg}")

    model = MVT(renderer_device=renderer_device, **mvt_cfg)
    return exp_cfg, mvt_cfg, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer-device", default="cuda:0")
    parser.add_argument("--mvt-cfg-path", default="mvt/configs/rvt2.yaml")
    args = parser.parse_args()

    _, official_mvt_cfg, official = build(
        "configs/rvt2.yaml", args.mvt_cfg_path, args.renderer_device
    )
    assert official_mvt_cfg.model == "MVTSingle"
    assert type(official.mvt1).__name__ == "MVT"
    assert official.num_img == 3

    exp_cfg, r3m_mvt_cfg, r3m = build(
        "configs/unadaptedR3M.yaml", args.mvt_cfg_path, args.renderer_device
    )
    assert r3m_mvt_cfg.model == "MVT_Resnet"
    assert type(r3m.mvt1).__name__ == "MVT_Resnet"
    assert r3m.mvt1.depth == 1
    assert r3m.pretrain_path == exp_cfg.pretrain
    assert r3m.mvt1.pretrain_path == exp_cfg.pretrain
    assert r3m_mvt_cfg.rend_three_views
    assert r3m.num_img == 3

    print("official_model=MVTSingle")
    print(f"official_num_views={official.num_img}")
    print("r3m_model=MVT_Resnet")
    print(f"r3m_depth={r3m.mvt1.depth}")
    print(f"r3m_pretrain={r3m.mvt1.pretrain_path}")
    print(f"r3m_num_views={r3m.num_img}")


if __name__ == "__main__":
    main()
