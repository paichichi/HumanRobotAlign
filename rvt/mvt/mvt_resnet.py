# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from math import ceil

import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat

import rvt.mvt.utils as mvt_utils
from rvt.mvt.attn import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    cache_fn,
    DenseBlock,
    FeedForward,
    FixedPositionalEncoding,
)
from rvt.mvt.raft_utils import ConvexUpSample

from .resnet import *


#### using no_ds resnet feature, only rgb, no proprio, 
#### directly patchify resnet50 feature into D dim tokens;
#### no intra-image self-attn,
class MVT_Resnet(nn.Module):
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
        norm_corr,
        add_pixel_loc,
        add_depth,
        pe_fix,
        renderer_device="cuda:0",
        renderer=None,
        adapter=[],
        ds_rate=1,
        output_dim=256,
        pretrain_path=None,
        rot_ver=0,
        feat_ver=0,
        cvx_up=False,
        xops=False,
        wpt_img_aug=0.0,
        num_rot=72,
        use_point_renderer=False,
        no_feat=False,
        **kwargs,
    ):
        """MultiView Transfomer

        :param depth: depth of the attention network
        :param img_size: number of pixels per side for rendering
        :param renderer_device: device for placing the renderer
        :param add_proprio:
        :param proprio_dim:
        :param add_lang:
        :param lang_dim:
        :param lang_len:
        :param img_feat_dim:
        :param feat_dim:
        :param im_channels: intermediate channel size
        :param attn_dim:
        :param attn_heads:
        :param attn_dim_head:
        :param activation:
        :param weight_tie_layers:
        :param attn_dropout:
        :param decoder_dropout:
        :param img_patch_size: intial patch size
        :param final_dim: final dimensions of features
        :param self_cross_ver:
        :param add_corr:
        :param add_pixel_loc:
        :param add_depth:
        :param pe_fix: matter only when add_lang is True
            Either:
                True: use position embedding only for image tokens
                False: use position embedding for lang and image token
        """

        super().__init__()
        if kwargs:
            raise TypeError(f"Unsupported MVT_Resnet options: {sorted(kwargs)}")
        if rot_ver not in (0, 1):
            raise NotImplementedError(
                "MVT_Resnet currently supports only rot_ver=0 or rot_ver=1."
            )
        if feat_ver not in (0, 1):
            raise NotImplementedError(
                "MVT_Resnet currently supports only feat_ver=0 or feat_ver=1."
            )
        self.cvx_up = cvx_up
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        self.lang_dim = lang_dim
        self.lang_len = lang_len
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.norm_corr = norm_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.pe_fix = pe_fix
        self.attn_dim = attn_dim
        self.use_point_renderer = use_point_renderer
        self.xops = xops
        self.feat_ver = feat_ver
        self.rot_ver = rot_ver
        self.wpt_img_aug = wpt_img_aug
        self.num_rot = num_rot
        self.pretrain_path = pretrain_path
        self.no_feat = no_feat

        self.adapter=adapter
        self.ds_rate=ds_rate
        print(
            "MVT_Resnet: "
            f"depth={self.depth}, img_size={self.img_size}, patch={self.img_patch_size}, "
            f"attn_dim={self.attn_dim}, ds_rate={self.ds_rate}, adapter={self.adapter}, "
            f"xops={self.xops}, feat_ver={self.feat_ver}, rot_ver={self.rot_ver}, "
            f"cvx_up={self.cvx_up}, point_renderer={self.use_point_renderer}, "
            f"pretrain={self.pretrain_path}"
        )
        self.convnet = resnet50(pretrained=None, adapter=self.adapter, ds_rate=self.ds_rate)
        self.convnet.fc = nn.Identity()
        for name, param in self.named_parameters():
            param.requires_grad = False
        

        assert not renderer is None
        self.renderer = renderer
        self.num_img = self.renderer.num_img

        # patchified input dimensions
        spatial_size = int(self.img_size*ds_rate) // self.img_patch_size  # 220 / 11 = 20

        # if self.add_proprio:
        #     # 64 img features + 64 proprio features
        #     self.input_dim_before_seq = self.im_channels * 2
        # else:
        #     self.input_dim_before_seq = self.im_channels

        self.custom_input_dim=2048
        self.input_dim_before_seq = output_dim

        # learnable positional encoding
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        if self.pe_fix:
            num_pe_token = spatial_size**2 * self.num_img
        else:
            num_pe_token = lang_max_seq_len + (spatial_size**2 * self.num_img)
        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.attn_dim, #self.input_dim_before_seq,
            )
        )

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1

        # img input preprocessing encoder
        self.input_preprocess = Conv2DBlock(
            self.custom_input_dim, #inp_img_feat_dim,
            self.input_dim_before_seq, #self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=activation,
        )
        # inp_pre_out_dim = self.im_channels

        if self.add_proprio:
            self.proprio_preprocess = DenseBlock(
                self.proprio_dim,
                self.attn_dim,
                norm="group",
                activation=activation,
            )

        self.patchify = Conv2DBlock(
            self.input_dim_before_seq, #self.custom_input_dim,
            attn_dim, #self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )

        # lang preprocess
        if self.add_lang:
            self.lang_preprocess = DenseBlock(
                lang_emb_dim,
                self.attn_dim, #self.input_dim_before_seq, #self.im_channels * 2,
                norm="group",
                activation=activation,
            )

        # self.fc_bef_attn = DenseBlock(
        #     self.input_dim_before_seq,
        #     attn_dim,
        #     norm=None,
        #     activation=None,
        # )
        self.fc_aft_attn = DenseBlock(
            attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            attn_dim,
            Attention(
                attn_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
                use_fast=xops,
            ),
        )
        get_attn_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))
        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        attn_depth = depth

        for _ in range(attn_depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)])
            )

        if self.cvx_up:
            self.up0 = ConvexUpSample(
                in_dim=self.input_dim_before_seq,
                out_dim=1,
                up_ratio=self.img_patch_size,
            )
        else:
            self.up0 = Conv2DUpsampleBlock(
                self.input_dim_before_seq,
                self.im_channels,
                kernel_sizes=self.img_patch_size,
                strides=self.img_patch_size,
                norm=None,
                activation=activation,
                out_size=(spatial_size * self.img_patch_size, spatial_size * self.img_patch_size),
            )

            #final_inp_dim = self.im_channels + self.custom_input_dim #inp_pre_out_dim
            #final_inp_dim = self.input_dim_before_seq + self.custom_input_dim 
            final_inp_dim = self.im_channels + self.input_dim_before_seq 

            if self.ds_rate!=1:
                self.up1 = nn.ConvTranspose2d(
                    in_channels=final_inp_dim, 
                    out_channels=im_channels, 
                    kernel_size=int(1/self.ds_rate)+1, #3, 
                    stride=int(1/self.ds_rate), #2,
                    padding=1, 
                    output_padding=1)
                final_inp_dim = im_channels

            # final layers
            self.final = Conv2DBlock(
                final_inp_dim,
                self.im_channels,
                kernel_sizes=3,
                strides=1,
                norm=None,
                activation=activation,
            )

            self.trans_decoder = Conv2DBlock(
                self.im_channels, #self.final_dim,
                1,
                kernel_sizes=3,
                strides=1,
                norm=None,
                activation=None,
            )

        feat_out_size = feat_dim
        feat_fc_dim = 0
        feat_fc_dim += self.input_dim_before_seq
        if self.cvx_up:
            feat_fc_dim += self.input_dim_before_seq
        else:
            feat_fc_dim += self.im_channels #self.final_dim

        def get_feat_fc(_feat_in_size, _feat_out_size, _feat_fc_dim=feat_fc_dim):
            return nn.Sequential(
                nn.Linear(_feat_in_size, _feat_fc_dim),
                nn.ReLU(),
                nn.Linear(_feat_fc_dim, _feat_fc_dim // 2),
                nn.ReLU(),
                nn.Linear(_feat_fc_dim // 2, _feat_out_size),
            )

        if self.rot_ver == 0:
            self.feat_fc = get_feat_fc(self.num_img * feat_fc_dim, feat_out_size)
        elif self.rot_ver == 1:
            assert self.num_rot * 3 <= feat_out_size
            feat_out_size_ex_rot = feat_out_size - (self.num_rot * 3)
            if feat_out_size_ex_rot > 0:
                self.feat_fc_ex_rot = get_feat_fc(
                    self.num_img * feat_fc_dim, feat_out_size_ex_rot
                )

            self.feat_fc_init_bn = nn.BatchNorm1d(self.num_img * feat_fc_dim)
            self.feat_fc_pe = FixedPositionalEncoding(
                self.num_img * feat_fc_dim, feat_scale_factor=1
            )
            self.feat_fc_x = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
            self.feat_fc_y = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
            self.feat_fc_z = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
        else:
            assert False

        if self.use_point_renderer:
            from point_renderer.rvt_ops import select_feat_from_hm
        else:
            from mvt.renderer import select_feat_from_hm
        global select_feat_from_hm

    def train(self, mode: bool = True):
        super().train(mode)
        self.convnet.eval()
        return self

    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """
        transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """
        pt_img = self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )
        return pt_img

    def forward(
        self,
        img,
        proprio=None,
        lang_emb=None,
        wpt_local=None,
        rot_x_y=None,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param wpt_local: gt waypoint location, used by feat_ver=1 while training
        :param rot_x_y: gt x/y rotation classes, used by rot_ver=1 while training
        :param img_aug: (float) magnitude of augmentation in rgb image
        """

        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_img == self.num_img
        # assert img_feat_dim == self.img_feat_dim
        assert h == w == self.img_size

        img = img.view(bs * num_img, img_feat_dim, h, w)
        ####### only using rgb
        img = img[:, 3:6, :, :].contiguous()
        

        with torch.no_grad():
            d0 = self.convnet(img)
        
        vis_feat=(torch.max(torch.max(d0,dim=-1,keepdim=False)[0],dim=-1,keepdim=False)[0]).view(bs, num_img, -1)
        vis_feat=torch.max(vis_feat,dim=-2,keepdim=False)[0]


        # print(d0.size())
        # preprocess
        # (bs * num_img, im_channels, h, w)
        ##### not using, but get the resnet feature
        # d0 = self.input_preprocess(img) 
        d0 = self.input_preprocess(d0) 
        
        _, _, h_dim, w_dim = d0.size()
        num_pat_img = h_dim // self.img_patch_size


        # (bs * num_img, im_channels, h, w) ->
        # (bs * num_img, im_channels, h / img_patch_strid, w / img_patch_strid) patches
        ###### directly patchifg the resnet feature map;
        ins = self.patchify(d0)
        # (bs, im_channels, num_img, h / img_patch_strid, w / img_patch_strid) patches
        ins = (
            ins.view(
                bs,
                num_img,
                self.attn_dim, #self.input_dim_before_seq, #self.im_channels,
                num_pat_img,
                num_pat_img,
            )
            .transpose(1, 2)
            .clone()
        )
        # print(ins.size())

        if self.add_proprio:
            assert proprio is not None
            p = self.proprio_preprocess(proprio)
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            ins = ins + p

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B, num_img, np, np, 128]

        # save original shape of input for layer
        ins_orig_shape = ins.shape

        # flatten patches into sequence
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, 128]
        # add learable pos encoding
        # only added to image tokens
        if self.pe_fix:
            ins += self.pos_encoding

        # append language features as sequence
        num_lang_tok = 0
        if self.add_lang:
            l = self.lang_preprocess(
                lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
            )
            l = l.view(bs, self.lang_max_seq_len, -1)
            num_lang_tok = l.shape[1]
            ins = torch.cat((l, ins), dim=1)  # [B, 77 + num_img * np * np, 128]

        # add learable pos encoding
        if not self.pe_fix:
            ins = ins + self.pos_encoding

        # x = self.fc_bef_attn(ins)
        x=ins

        if self.self_cross_ver == 0:
            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        elif self.self_cross_ver == 1:
            # lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]

            # # within image self attention
            # imgx = imgx.reshape(bs * num_img, num_pat_img * num_pat_img, -1)
            # for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
            #     imgx = self_attn(imgx) + imgx
            #     imgx = self_ff(imgx) + imgx

            # imgx = imgx.view(bs, num_img * num_pat_img * num_pat_img, -1)
            # x = torch.cat((lx, imgx), dim=1)
            # # cross attention
            # for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
            #     x = self_attn(x) + x
            #     x = self_ff(x) + x
            
            # cross attention
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        else:
            assert False

        # append language features as sequence
        if self.add_lang:
            # throwing away the language embeddings
            x = x[:, num_lang_tok:]
        
        x = self.fc_aft_attn(x)

        # reshape back to orginal size
        x = x.view(bs, *ins_orig_shape[1:-1], x.shape[-1])  # [B, num_img, np, np, 128]
        x = rearrange(x, "b ... d -> b d ...")  # [B, 128, num_img, np, np]

        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        _feat = _feat.view(bs, -1)
        feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.input_dim_before_seq, num_pat_img, num_pat_img
            )
        )

        if self.cvx_up:
            trans = self.up0(x).view(bs, self.num_img, h, w)
        else:
            u0 = self.up0(x)
            u0 = torch.cat([u0, d0], dim=1)
            if self.ds_rate!=1:
                u0 = self.up1(u0)  ### add to upsample the downsampled resnet feature

            u = self.final(u0)

            # translation decoder
            trans = self.trans_decoder(u).view(bs, self.num_img, h, w)

        if self.no_feat:
            return {"trans": trans, "vis_feat": vis_feat}

        if self.feat_ver == 0:
            hm = F.softmax(trans.detach().view(bs, self.num_img, h * w), 2).view(
                bs * self.num_img, 1, h, w
            )
            if self.cvx_up:
                _hm = F.unfold(
                    hm,
                    kernel_size=self.img_patch_size,
                    padding=0,
                    stride=self.img_patch_size,
                )
                assert _hm.shape == (
                    bs * self.num_img,
                    self.img_patch_size * self.img_patch_size,
                    num_pat_img * num_pat_img,
                )
                _hm = torch.mean(_hm, 1)
                _hm = _hm.view(bs * self.num_img, 1, num_pat_img, num_pat_img)
                _u = x
            else:
                _hm = hm
                _u = u

            _feat = torch.sum(_hm * _u, dim=[2, 3])
            _feat = _feat.view(bs, -1)
        elif self.feat_ver == 1:
            if self.training:
                assert wpt_local is not None
            else:
                wpt_local = self.get_wpt(
                    out={"trans": trans.clone().detach()},
                    dyn_cam_info=None,
                )

            wpt_img = self.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                dyn_cam_info=None,
            )
            wpt_img = wpt_img.reshape(bs * self.num_img, 2)

            if self.training:
                wpt_img = mvt_utils.add_uni_noi(
                    wpt_img, self.wpt_img_aug * self.img_size
                )
                wpt_img = torch.clamp(wpt_img, 0, self.img_size - 1)

            if self.cvx_up:
                _wpt_img = wpt_img / self.img_patch_size
                _u = x
                assert (
                    0 <= _wpt_img.min() and _wpt_img.max() <= x.shape[-1]
                ), print(_wpt_img, x.shape)
            else:
                _wpt_img = wpt_img
                _u = u

            _feat = select_feat_from_hm(_wpt_img.unsqueeze(1), _u)[0]
            _feat = _feat.view(bs, -1)
        else:
            assert False

        feat.append(_feat)
        feat = torch.cat(feat, dim=-1)
        if self.rot_ver == 0:
            feat = self.feat_fc(feat)
            out = {"feat": feat}
        elif self.rot_ver == 1:
            assert rot_x_y is not None or not self.training
            feat_ex_rot = self.feat_fc_ex_rot(feat)

            feat_rot = self.feat_fc_init_bn(feat)
            feat_x = self.feat_fc_x(feat_rot)

            if self.training:
                rot_x = rot_x_y[..., 0].view(bs, 1)
            else:
                rot_x = feat_x.argmax(dim=1, keepdim=True)
            rot_x_pe = self.feat_fc_pe(rot_x)
            feat_y = self.feat_fc_y(feat_rot + rot_x_pe)

            if self.training:
                rot_y = rot_x_y[..., 1].view(bs, 1)
            else:
                rot_y = feat_y.argmax(dim=1, keepdim=True)
            rot_y_pe = self.feat_fc_pe(rot_y)
            feat_z = self.feat_fc_z(feat_rot + rot_x_pe + rot_y_pe)
            out = {
                "feat_ex_rot": feat_ex_rot,
                "feat_x": feat_x,
                "feat_y": feat_y,
                "feat_z": feat_z,
            }
        else:
            assert False

        out.update({"trans": trans, "vis_feat": vis_feat})

        return out

    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]

        q_trans = out["trans"].view(bs, nc, h * w)
        hm = torch.nn.functional.softmax(q_trans, 2)
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)
        if self.use_point_renderer:
            pred_wpt = pred_wpt.squeeze(1)

        assert y_q is None

        return pred_wpt

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()
