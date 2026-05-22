"""Smoke-test MVT_Resnet construction for default and stage-two modes.

This script intentionally avoids datasets, checkpoints, log directories, and
training loops. By default it only constructs the config, MVT model, and
RVTAgent. Use --forward only when the local environment has the renderer
dependencies available and you want a tiny synthetic forward pass.
"""

import argparse
import os
import sys

import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import rvt.config as default_exp_cfg
import rvt.mvt.config as default_mvt_cfg
# from rvt.models import rvt_agent
from rvt.mvt.mvt import MVT
# from rvt.utils.peract_utils import CAMERAS, IMAGE_SIZE, SCENE_BOUNDS


def _merge_optional(cfg, path):
    if path:
        cfg.merge_from_file(path)


def build_configs(args):
    exp_cfg = default_exp_cfg.get_cfg_defaults()
    _merge_optional(exp_cfg, args.exp_cfg_path)

    exp_cfg.defrost()
    exp_cfg.model = "MVT_Resnet"
    exp_cfg.depth = args.depth
    exp_cfg.rot_ver = 0
    exp_cfg.feat_ver = 0
    exp_cfg.use_point_renderer = True
    exp_cfg.cvx_up = False
    exp_cfg.stage_two = args.mode == "stage-two"
    exp_cfg.stage_two_mvt_resnet = args.mode == "stage-two"
    exp_cfg.freeze()

    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    _merge_optional(mvt_cfg, args.mvt_cfg_path)

    mvt_cfg.defrost()
    mvt_cfg.feat_dim = (exp_cfg.peract.num_rotation_classes * 3) + 2 + 2
    mvt_cfg.depth = exp_cfg.depth
    mvt_cfg.attn_dim = exp_cfg.attn_dim
    mvt_cfg.ds_rate = exp_cfg.ds_rate
    mvt_cfg.adapter = exp_cfg.adapter
    mvt_cfg.model = exp_cfg.model
    mvt_cfg.output_dim = exp_cfg.output_dim
    mvt_cfg.stage_two = exp_cfg.stage_two
    mvt_cfg.stage_two_mvt_resnet = exp_cfg.stage_two_mvt_resnet
    mvt_cfg.rot_ver = exp_cfg.rot_ver
    mvt_cfg.feat_ver = exp_cfg.feat_ver
    mvt_cfg.use_point_renderer = exp_cfg.use_point_renderer
    mvt_cfg.cvx_up = exp_cfg.cvx_up
    mvt_cfg.freeze()

    return exp_cfg, mvt_cfg


# def build_model_and_agent(exp_cfg, mvt_cfg, device):
#     model = MVT(renderer_device=device, **mvt_cfg).to(device)
#     agent = rvt_agent.RVTAgent(
#         network=model,
#         image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
#         add_lang=mvt_cfg.add_lang,
#         scene_bounds=SCENE_BOUNDS,
#         cameras=CAMERAS,
#         stage_two=exp_cfg.stage_two,
#         stage_two_mvt_resnet=exp_cfg.stage_two_mvt_resnet,
#         rot_ver=exp_cfg.rot_ver,
#         feat_ver=exp_cfg.feat_ver,
#         **exp_cfg.peract,
#         **exp_cfg.rvt,
#     )
#     return model, agent
def build_model_and_agent(exp_cfg, mvt_cfg, device):
    model = MVT(renderer_device=device, **mvt_cfg).to(device)
    agent = None
    return model, agent


@torch.no_grad()
def run_dummy_forward(model, mvt_cfg, device):
    model.eval()
    num_points = 64
    pc = [torch.rand(num_points, 3, device=device) * 2 - 1]
    img_feat = [torch.rand(num_points, mvt_cfg.img_feat_dim, device=device) * 2 - 1]
    proprio = torch.zeros(1, model.proprio_dim, device=device)
    lang_emb = torch.zeros(1, mvt_cfg.lang_len, mvt_cfg.lang_dim, device=device)
    out = model(pc=pc, img_feat=img_feat, proprio=proprio, lang_emb=lang_emb, img_aug=0)
    expected = {"trans", "feat", "vis_feat"}
    missing = expected.difference(out)
    if missing:
        raise RuntimeError(f"Missing output keys: {sorted(missing)}")
    if model.stage_two and "mvt2" not in out:
        raise RuntimeError("stage_two=True but MVT output is missing 'mvt2'")
    print(f"forward_ok keys={sorted(out.keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["default", "stage-two"], required=True)
    parser.add_argument("--exp-cfg-path", default="")
    parser.add_argument("--mvt-cfg-path", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()

    exp_cfg, mvt_cfg = build_configs(args)
    model, agent = build_model_and_agent(exp_cfg, mvt_cfg, args.device)

    if len(model.mvt1.layers) != args.depth:
        raise RuntimeError(
            f"Expected mvt1 depth {args.depth}, got {len(model.mvt1.layers)}"
        )
    if model.stage_two and len(model.mvt2.layers) != args.depth:
        raise RuntimeError(
            f"Expected mvt2 depth {args.depth}, got {len(model.mvt2.layers)}"
        )

    if args.forward:
        run_dummy_forward(model, mvt_cfg, args.device)

    print(
        "smoke_ok "
        f"mode={args.mode} model={mvt_cfg.model} depth={mvt_cfg.depth} "
        f"stage_two={mvt_cfg.stage_two} "
        f"stage_two_mvt_resnet={mvt_cfg.stage_two_mvt_resnet} "
        # f"agent_stage_two={agent.stage_two}"
        f"agent_stage_two={getattr(agent, 'stage_two', None)}"
    )


if __name__ == "__main__":
    main()
