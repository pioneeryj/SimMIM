# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer

# SwinTransformerForSimMIM 클래스 정의: SwinTransformer를 상속받아 SimMIM에 맞게 수정
class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 부모 클래스인 SwinTransformer의 초기화 호출

        assert self.num_classes == 0  # num_classes는 0이어야 함 (클래스 분류가 아닌 복원 문제이므로)

        # 마스크 토큰 초기화 (임베딩 차원 크기로 정의)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)  # 정규분포로 마스크 토큰 초기화

    def forward(self, x, mask):
        # 입력 이미지를 패치로 나누고 각 패치를 임베딩
        x = self.patch_embed(x)

        assert mask is not None  # 마스크가 반드시 필요함
        B, L, _ = x.shape  # 배치 크기(B), 패치 수(L), 임베딩 차원(C)

        # 마스크 토큰을 확장하여 마스크된 위치에 적용
        mask_tokens = self.mask_token.expand(B, L, -1)  # 마스크 토큰을 배치 크기와 패치 수에 맞게 확장
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)  # 마스크 형태 변경 (B, L, 1)
        x = x * (1. - w) + mask_tokens * w  # 마스크된 위치에 마스크 토큰 삽입

        # 절대 위치 임베딩 추가 (선택적)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)  # 드롭아웃 적용

        # Swin Transformer 레이어들을 통과
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # 정규화 적용

        # 임베딩을 2D 형태로 변환 (이미지 형태로 복원)
        x = x.transpose(1, 2)  # (B, C, L) 형태로 변환
        B, C, L = x.shape
        H = W = int(L ** 0.5)  # 가로, 세로 크기 계산 (정사각형 이미지 가정)
        x = x.reshape(B, C, H, W)  # (B, C, H, W) 형태로 변환
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        # 마스크 토큰에는 weight decay를 적용하지 않음
        return super().no_weight_decay() | {'mask_token'}


# VisionTransformerForSimMIM 클래스 정의: VisionTransformer를 상속받아 SimMIM에 맞게 수정
class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0  # num_classes는 0이어야 함 (복원 문제이므로)

        # 마스크 토큰 초기화
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)  # 정규분포로 마스크 토큰 초기화

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        # 정규분포로 텐서 초기화
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        # 입력 이미지를 패치로 나누고 각 패치를 임베딩
        x = self.patch_embed(x)

        assert mask is not None  # 마스크가 반드시 필요함
        B, L, _ = x.shape  # 배치 크기(B), 패치 수(L), 임베딩 차원(C)

        # 마스크 토큰을 확장하여 마스크된 위치에 적용
        mask_token = self.mask_token.expand(B, L, -1)  # 마스크 토큰을 배치 크기와 패치 수에 맞게 확장
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)  # 마스크 형태 변경 (B, L, 1)
        x = x * (1 - w) + mask_token * w  # 마스크된 위치에 마스크 토큰 삽입

        # 클래스 토큰 추가
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 클래스 토큰을 배치 크기만큼 확장 (B, 1, C)
        x = torch.cat((cls_tokens, x), dim=1)  # 클래스 토큰을 패치 임베딩에 결합 (B, L+1, C)

        # 위치 임베딩 추가 (선택적)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)  # 드롭아웃 적용

        # Transformer 블록을 통과하며 처리
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)  # 정규화 적용

        # 클래스 토큰 제외하고 나머지 복원
        x = x[:, 1:]  # 첫 번째 위치의 클래스 토큰 제거
        B, L, C = x.shape
        H = W = int(L ** 0.5)  # 가로, 세로 크기 계산 (정사각형 이미지 가정)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W) 형태로 변환
        return x


# SimMIM 클래스 정의: 전체 모델 구성 (Encoder + Decoder)
class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder  # 인코더 (Swin 또는 Vision Transformer)
        self.encoder_stride = encoder_stride  # 인코더의 스트라이드 값

        # 디코더 정의 (인코더 출력을 원본 이미지로 복원)
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,  # 인코더 출력 채널 수
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),  # 출력 채널 수는 원본 이미지의 채널 수 * 패치 크기 제곱
            nn.PixelShuffle(self.encoder_stride),  # 픽셀 셔플을 통해 공간 해상도 복원
        )

        self.in_chans = self.encoder.in_chans  # 입력 채널 수 (예: RGB 이미지면 3)
        self.patch_size = self.encoder.patch_size  # 패치 크기

    def forward(self, x, mask):
        # 인코더를 통해 마스크된 이미지를 인코딩
        z = self.encoder(x, mask)
        # 디코더를 통해 원본 이미지 복원
        x_rec = self.decoder(z)

        # 마스크를 이미지 크기에 맞게 확장 (원본 이미지의 패치 크기에 맞게)
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        # 복원 손실 계산 (L1 손실 사용)
        loss_recon = F.l1_loss(x, x_rec, reduction='none')  # 각 위치별 복원 손실 계산
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans  # 마스크된 위치의 평균 손실 계산
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        # 인코더의 특정 파라미터에 대해 weight decay를 적용하지 않음
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        # 인코더의 특정 키워드에 대해 weight decay를 적용하지 않음
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


# SimMIM 모델을 빌드하는 함수
def build_simmim(config):
    model_type = config.MODEL.TYPE
    # Swin Transformer 기반의 인코더 설정
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,  # 입력 이미지 크기
            patch_size=config.MODEL.SWIN.PATCH_SIZE,  # 패치 크기
            in_chans=config.MODEL.SWIN.IN_CHANS,  # 입력 채널 수
            num_classes=0,  # 클래스 수는 0 (복원 문제이므로)
            embed_dim=config.MODEL.SWIN.EMBED_DIM,  # 임베딩 차원 수
            depths=config.MODEL.SWIN.DEPTHS,  # 각 스테이지의 깊이
            num_heads=config.MODEL.SWIN.NUM_HEADS,  # 각 레이어의 헤드 수
            window_size=config.MODEL.SWIN.WINDOW_SIZE,  # 윈도우 크기
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,  # MLP 비율
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,  # QKV 바이어스 사용 여부
            qk_scale=config.MODEL.SWIN.QK_SCALE,  # QK 스케일링 값
            drop_rate=config.MODEL.DROP_RATE,  # 드롭아웃 비율
            drop_path_rate=config.MODEL.DROP_PATH_RATE,  # 드롭 패스 비율
            ape=config.MODEL.SWIN.APE,  # 절대 위치 임베딩 사용 여부
            patch_norm=config.MODEL.SWIN.PATCH_NORM,  # 패치 정규화 사용 여부
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)  # 체크포인트 사용 여부
        encoder_stride = 32  # 인코더 스트라이드 설정
    # Vision Transformer 기반의 인코더 설정
    elif model_type == 'vit':
        encoder = VisionTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,  # 입력 이미지 크기
            patch_size=config.MODEL.VIT.PATCH_SIZE,  # 패치 크기
            in_chans=config.MODEL.VIT.IN_CHANS,  # 입력 채널 수
            num_classes=0,  # 클래스 수는 0 (복원 문제이므로)
            embed_dim=config.MODEL.VIT.EMBED_DIM,  # 임베딩 차원 수
            depth=config.MODEL.VIT.DEPTH,  # Transformer 블록의 깊이
            num_heads=config.MODEL.VIT.NUM_HEADS,  # 각 레이어의 헤드 수
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,  # MLP 비율
            qkv_bias=config.MODEL.VIT.QKV_BIAS,  # QKV 바이어스 사용 여부
            drop_rate=config.MODEL.DROP_RATE,  # 드롭아웃 비율
            drop_path_rate=config.MODEL.DROP_PATH_RATE,  # 드롭 패스 비율
            norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 정규화 레이어 설정
            init_values=config.MODEL.VIT.INIT_VALUES,  # 초기 값 설정
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,  # 절대 위치 임베딩 사용 여부
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,  # 상대 위치 바이어스 사용 여부
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,  # 공유 상대 위치 바이어스 사용 여부
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)  # 평균 풀링 사용 여부
        encoder_stride = 16  # 인코더 스트라이드 설정
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    # SimMIM 모델 생성
    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)

    return model
