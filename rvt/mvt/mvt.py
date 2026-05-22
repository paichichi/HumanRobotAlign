# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import copy
import torch

from torch import nn

import rvt.mvt.utils as mvt_utils

from rvt.mvt.mvt_single import MVT as MVTSingle
from rvt.mvt.config import get_cfg_defaults
# from rvt.mvt.renderer import BoxRenderer
try:
    from rvt.mvt.renderer import BoxRenderer
except ModuleNotFoundError as e:
    if e.name == "pytorch3d":
        BoxRenderer = None
    else:
        raise

from rvt.mvt.mvt_resnet import *


class MVT(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        add_proprio,
        proprio_dim,
        add_lang,
        lang_dim,
        lang_len,
        img_feat_dim,
        feat_dim,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        add_pixel_loc,
        add_depth,
        pe_fix,
        renderer_device="cuda:0",
        model="MVTSingle",
        adapter=[],
        ds_rate=1,
        output_dim=256,
        stage_two=False,
        stage_two_mvt_resnet=False,
        rot_ver=0,
        num_rot=72,
        rot_x_y_aug=2,
        feat_ver=0,
        use_point_renderer=False,
        cvx_up=False,
        rend_three_views=False,
        norm_corr=False,
        inp_pre_pro=True,
        inp_pre_con=True,
        wpt_img_aug=0.01,
        st_sca=4,
        st_wpt_loc_aug=0.05,
        st_wpt_loc_inp_no_noise=False,
        img_aug_2=0.0,
    ):
        """MultiView Transfomer"""
        super().__init__()

        if stage_two and not (model == "MVT_Resnet" and stage_two_mvt_resnet):
            raise NotImplementedError(
                "stage_two=True is supported only for the experimental "
                "MVT_Resnet path. Set stage_two_mvt_resnet: true with "
                "model: 'MVT_Resnet', or keep stage_two: false."
            )
        if rot_ver != 0:
            raise NotImplementedError(
                "rot_ver=1 is not supported for MVT_Resnet in Phase 2. "
                "Keep rot_ver: 0 until feat_x/feat_y/feat_z/feat_ex_rot heads are added."
            )
        if feat_ver != 0:
            raise NotImplementedError(
                "feat_ver=1 is not supported for MVT_Resnet in Phase 2. "
                "Keep feat_ver: 0 until waypoint-conditioned feature extraction is added."
            )
        if use_point_renderer:
            raise NotImplementedError(
                "use_point_renderer=True is not supported in this HR-Align Phase 1 path. "
                "Keep use_point_renderer: false."
            )
        if cvx_up:
            raise NotImplementedError(
                "cvx_up=True is not supported in this HR-Align Phase 1 path. "
                "Keep cvx_up: false."
            )
        if rend_three_views:
            raise NotImplementedError(
                "rend_three_views=True is not supported in this HR-Align Phase 1 path. "
                "Keep rend_three_views: false."
            )
        if norm_corr:
            raise NotImplementedError(
                "norm_corr=True is not supported in this HR-Align Phase 2 path. "
                "Keep norm_corr: false."
            )
        if img_aug_2 != 0:
            raise NotImplementedError(
                "img_aug_2 is not supported in this HR-Align Phase 2 path. "
                "Keep img_aug_2: 0.0."
            )

        # creating a dictonary of all the input parameters
        args = copy.deepcopy(locals())
        del args["self"]
        del args["__class__"]
        
        del args["model"]
        del args["stage_two"]
        del args["stage_two_mvt_resnet"]
        del args["rot_ver"]
        del args["num_rot"]
        del args["rot_x_y_aug"]
        del args["feat_ver"]
        del args["use_point_renderer"]
        del args["cvx_up"]
        del args["rend_three_views"]
        del args["norm_corr"]
        del args["inp_pre_pro"]
        del args["inp_pre_con"]
        del args["wpt_img_aug"]
        del args["st_sca"]
        del args["st_wpt_loc_aug"]
        del args["st_wpt_loc_inp_no_noise"]
        del args["img_aug_2"]

        self.stage_two = stage_two
        self.stage_two_mvt_resnet = stage_two_mvt_resnet
        self.st_sca = st_sca
        self.st_wpt_loc_aug = st_wpt_loc_aug
        self.st_wpt_loc_inp_no_noise = st_wpt_loc_inp_no_noise

        # for verifying the input
        self.img_feat_dim = img_feat_dim
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            with_depth=add_depth,
        )
        self.num_img = self.renderer.num_img
        self.proprio_dim = proprio_dim
        self.img_size = img_size


        #self.mvt1 = MVTSingle(**args, renderer=self.renderer)

        if model=="MVTSingle":
            mvt_cls = MVTSingle
        elif model=="MVT_Resnet":
            mvt_cls = MVT_Resnet
        else:
            raise ValueError(f"Unsupported MVT model: {model}")

        self.mvt1 = mvt_cls(**args, renderer=self.renderer)
        if self.stage_two:
            self.mvt2 = mvt_cls(**args, renderer=self.renderer)

    def get_pt_loc_on_img(self, pt, dyn_cam_info, out=None, mvt1_or_mvt2=True):
        """
        :param pt: point for which location on image is to be found. the point
            shoud be in the same reference frame as wpt_local (see forward()),
            even for mvt2
        :param out: output from mvt, when using mvt2, we also need to provide the
            origin location where where the point cloud needs to be shifted
            before estimating the location in the image
        """
        assert len(pt.shape) == 3
        bs, np, x = pt.shape
        assert x == 3
        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            assert out is None
            out = self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)
        else:
            assert self.stage_two
            assert out is not None
            assert out["wpt_local1"].shape == (bs, 3)
            pt = self.st_sca * (pt - out["wpt_local1"].unsqueeze(1))
            pt = pt.view(bs, np, 3)
            out = self.mvt2.get_pt_loc_on_img(pt, dyn_cam_info)

        return out

    def get_wpt(self, out, dyn_cam_info, y_q=None, mvt1_or_mvt2=True):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        :param y_q: refer to the definition in mvt_single.get_wpt
        """
        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            wpt = self.mvt1.get_wpt(out, dyn_cam_info, y_q)
        else:
            assert self.stage_two
            wpt = self.mvt2.get_wpt(out["mvt2"], dyn_cam_info, y_q)
            wpt = out["rev_trans"](wpt)
        return wpt

    def render(self, pc, img_feat, img_aug, dyn_cam_info, mvt1_or_mvt2=True):
        assert isinstance(mvt1_or_mvt2, bool)
        mvt = self.mvt1 if mvt1_or_mvt2 else self.mvt2

        with torch.no_grad():
            if dyn_cam_info is None:
                dyn_cam_info_itr = (None,) * len(pc)
            else:
                dyn_cam_info_itr = dyn_cam_info

            if mvt.add_corr:
                img = [
                    self.renderer(
                        _pc,
                        torch.cat((_pc, _img_feat), dim=-1),
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,)
                        if not (_dyn_cam_info is None)
                        else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(
                        pc, img_feat, dyn_cam_info_itr
                    )
                ]
            else:
                img = [
                    self.renderer(
                        _pc,
                        _img_feat,
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,)
                        if not (_dyn_cam_info is None)
                        else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(
                        pc, img_feat, dyn_cam_info_itr
                    )
                ]

            img = torch.cat(img, 0)
            img = img.permute(0, 1, 4, 2, 3)

            # for visualization purposes
            if mvt.add_corr:
                mvt.img = img[:, :, 3:].clone().detach()
            else:
                mvt.img = img.clone().detach()

            # image augmentation
            if img_aug != 0:
                stdv = img_aug * torch.rand(1, device=img.device)
                # values in [-stdv, stdv]
                noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
                img = torch.clamp(img + noise, -1, 1)

            if mvt.add_pixel_loc:
                bs = img.shape[0]
                pixel_loc = mvt.pixel_loc.to(img.device)
                img = torch.cat(
                    (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
                )

        return img

    def verify_inp(
        self,
        pc,
        img_feat,
        proprio,
        lang_emb,
        img_aug,
        wpt_local=None,
        rot_x_y=None,
    ):
        if not self.training:
            # no img_aug when not training
            assert img_aug == 0
            assert rot_x_y is None, f"rot_x_y={rot_x_y}"
        if self.training:
            if self.stage_two:
                assert wpt_local is not None, "stage_two training requires wpt_local"
            assert rot_x_y is None, f"rot_x_y={rot_x_y}"

        bs = len(pc)
        assert bs == len(img_feat)

        for _pc, _img_feat in zip(pc, img_feat):
            np, x1 = _pc.shape
            np2, x2 = _img_feat.shape

            assert np == np2
            assert x1 == 3
            assert x2 == self.img_feat_dim

        if self.add_proprio:
            bs3, x3 = proprio.shape
            assert bs == bs3
            assert (
                x3 == self.proprio_dim
            ), "Does not support proprio of shape {proprio.shape}"
        else:
            assert proprio is None, "Invalid input for proprio={proprio}"

        if self.add_lang:
            bs4, x4, x5 = lang_emb.shape
            assert bs == bs4
            assert (
                x4 == self.lang_max_seq_len
            ), "Does not support lang_emb of shape {lang_emb.shape}"
            assert (
                x5 == self.lang_emb_dim
            ), "Does not support lang_emb of shape {lang_emb.shape}"
        else:
            assert (lang_emb is None) or (
                torch.all(lang_emb == 0)
            ), f"Invalid input for lang={lang}"

        if wpt_local is not None:
            bs5, x6 = wpt_local.shape
            assert bs == bs5
            assert x6 == 3, "Does not support wpt_local of shape {wpt_local.shape}"

    def forward(
        self,
        pc,
        img_feat,
        proprio=None,
        lang_emb=None,
        img_aug=0,
        wpt_local=None,
        rot_x_y=None,
        **kwargs,
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape
            (bs, num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        """

        self.verify_inp(pc, img_feat, proprio, lang_emb, img_aug, wpt_local, rot_x_y)
        # bs, [Nx, 3], bs, [Nx,3]
        # print("input:",len(pc), pc[0].size(), len(img_feat), img_feat[0].size())
        img = self.render(
            pc,
            img_feat,
            img_aug,
            dyn_cam_info=None,
            mvt1_or_mvt2=True,
        )
        # print("input img:", img.size()) # [B, 5, 10, 220, 220]
        if self.training:
            wpt_local_stage_one = wpt_local.clone().detach() if wpt_local is not None else None
        else:
            wpt_local_stage_one = wpt_local
        out = self.mvt1(
            img=img,
            proprio=proprio,
            lang_emb=lang_emb,
            wpt_local=wpt_local_stage_one,
            rot_x_y=rot_x_y,
            **kwargs,
        )
        if self.stage_two:
            with torch.no_grad():
                if self.training:
                    wpt_local_stage_one_noisy = mvt_utils.add_uni_noi(
                        wpt_local_stage_one.clone().detach(), 2 * self.st_wpt_loc_aug
                    )
                    pc, rev_trans = mvt_utils.trans_pc(
                        pc, loc=wpt_local_stage_one_noisy, sca=self.st_sca
                    )
                    if self.st_wpt_loc_inp_no_noise:
                        wpt_local2, _ = mvt_utils.trans_pc(
                            wpt_local, loc=wpt_local_stage_one_noisy, sca=self.st_sca
                        )
                    else:
                        wpt_local2, _ = mvt_utils.trans_pc(
                            wpt_local, loc=wpt_local_stage_one, sca=self.st_sca
                        )
                else:
                    wpt_local_stage_one = self.get_wpt(
                        out, dyn_cam_info=None, y_q=None, mvt1_or_mvt2=True
                    )
                    pc, rev_trans = mvt_utils.trans_pc(
                        pc, loc=wpt_local_stage_one, sca=self.st_sca
                    )
                    wpt_local_stage_one_noisy = wpt_local_stage_one
                    wpt_local2 = None

                img = self.render(
                    pc,
                    img_feat,
                    img_aug,
                    dyn_cam_info=None,
                    mvt1_or_mvt2=False,
                )

            out_mvt2 = self.mvt2(
                img=img,
                proprio=proprio,
                lang_emb=lang_emb,
                wpt_local=wpt_local2,
                rot_x_y=rot_x_y,
                **kwargs,
            )
            out["wpt_local1"] = wpt_local_stage_one_noisy
            out["rev_trans"] = rev_trans
            out["mvt2"] = out_mvt2
        # dict, ['trans','feat'], [bs,5,220,220], [bs, 220]
        # print("output:",type(out), out.keys(), out['trans'].size(), out['feat'].size())
        # print("vis feat:", out['vis_feat'].size())  ### [bs*5, C, H, W]

        return out

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    mvt = MVT(**cfg)
    breakpoint()
