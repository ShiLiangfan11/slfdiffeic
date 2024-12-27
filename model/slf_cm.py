# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
from torch.autograd import Function
from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from compressai.ops import quantize_ste
from .common_model import CompressionModel
from .layers_fm import conv3x3, DepthConvBlock2, DepthConvBlock3, DepthConvBlock4, \
    ResidualBlockUpsample, ResidualBlockWithStride2,UNet,ResidualBlockWitho
from utils.stream_helper import write_ip, get_downsampled_shape
from utils.func import update_registered_buffers, get_scale_table

class IntraEncoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.enc_1 = nn.Sequential(
            ResidualBlockWitho(4, 32, inplace=inplace),
            DepthConvBlock3(32, 32, inplace=inplace),
        )
        self.enc_2 = nn.Sequential(
            ResidualBlockWitho(32, 48, inplace=inplace),
            DepthConvBlock3(48, 48, inplace=inplace),
            ResidualBlockWitho(48, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )

    def forward(self, x):
        out = self.enc_1(x)
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.dec_1 = nn.Sequential(
            DepthConvBlock3(N, N, inplace=inplace),
            ResidualBlockWitho(N, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            ResidualBlockWitho(N, 48, inplace=inplace),
            DepthConvBlock3(48, 48, inplace=inplace),
           ResidualBlockWitho(48, 32,  inplace=inplace),
        )
        self.dec_2 = nn.Sequential(
            DepthConvBlock3(32, 32, inplace=inplace),
            ResidualBlockUpsample(32, 16, 2, inplace=inplace),
        )

    def forward(self, x):
        out = self.dec_1(x)
        return self.dec_2(out)


class DMCI(CompressionModel):
    def __init__(self, N=64, z_channel=32, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='gaussian', z_channel=z_channel,
                         ec_thread=ec_thread, stream_part=stream_part)

        self.enc = IntraEncoder(N, inplace)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock4(N, z_channel, inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
        )
        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel, 2, inplace=inplace),
            ResidualBlockUpsample(z_channel, z_channel, 2, inplace=inplace),
            DepthConvBlock4(z_channel, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock4(N, N * 2, inplace=inplace),
            DepthConvBlock4(N * 2, N * 2 + 2, inplace=inplace),
        )

        self.y_spatial_prior_reduction = nn.Conv2d(N * 2 + 2, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock2(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock2(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock2(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
        )

        self.dec = IntraDecoder(N, inplace)
        self.refine = nn.Sequential(
            UNet(16, 16, inplace=inplace),
            conv3x3(16, 4),
        )
        self.entropy_bottleneck = EntropyBottleneck(z_channel)
        self.gaussian_conditional = GaussianConditional(None)
        self.y_q_basic = nn.Parameter(torch.ones((1, N, 1, 1)))



    def get_curr_y_q(self, q_scale):
        q_basic = LowerBound.apply(self.y_q_basic, 0.5)
        return q_basic * q_scale

    def forward(self, x, image ,qs_global):
        _, _, H, W = image.size()
        device = x.device
        q_g = self.get_curr_y_q(qs_global)
        y = self.enc(x)
        y = y / q_g
        y_pad, slice_shape = self.pad_for_y(y)
        
        z = self.hyper_enc(y_pad)
        _, z_likelihoods = self.entropy_bottleneck(z)
        _, q_z_likelihoods = self.entropy_bottleneck(z, False)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset
        
        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_res, y_q,y_q_sof, y_hat, scales_hat = self.forward_four_part_prior(
            y, params,
            self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
            y_spatial_prior_reduction=self.y_spatial_prior_reduction)
        _, y_likelihoods = self.gaussian_conditional(y_res, scales_hat, None)
        _, q_likelihoods = self.gaussian_conditional(y_res, scales_hat, None, False)
        y_hat = y_hat * q_g
        x_hat = self.dec(y_hat)
        output= self.refine(x_hat)
        return output, [y_likelihoods, z_likelihoods], [q_likelihoods, q_z_likelihoods]
    
    def encode(self, x, qs_global, sps_id=0, output_file=None):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        if output_file is None:
            encoded = self.forward(x, qs_global)
            result = {
                'bit': encoded['bit'].item(),
                'x_hat': encoded['x_hat'],
            }
            return result

        compressed = self.compress(x, qs_global)
        bit_stream = compressed['bit_stream']
        with open(output_file, "wb") as f:  # 使用 "wb" 模式以二进制写入
         written = write_ip(f, True, sps_id, bit_stream)  # 传递文件对象 f
        result = {
            'bit': written * 8,
            'x_hat': compressed['x_hat'],
        }
        return result

    def compress(self, x, qs_global):
        device = x.device
        q_g = self.get_curr_y_q(qs_global)


        y = self.enc(x)
        y = y / q_g
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        torch.backends.cudnn.deterministic = True
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
                y_spatial_prior_reduction=self.y_spatial_prior_reduction)

        self.entropy_coder.reset()
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        self.entropy_coder.flush()
        y_hat = y_hat * q_g
        output = self.dec(y_hat)

        x_hat = self.refine(output)
        bit_stream = self.entropy_coder.get_encoded_stream()

        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result

    def decompress(self, bit_stream, shape,qs_global):
        B,_,H,W = shape
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        q_g = self.get_curr_y_q(qs_global)
       

        self.entropy_coder.set_stream(bit_stream)
        z_size = get_downsampled_shape(H, W, 64)
        y_height, y_width = get_downsampled_shape(H, W, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        z_q = self.bit_estimator_z.decode_stream(z_size, dtype, device, qs_global)
        z_hat = z_q

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior,
                                                self.y_spatial_prior_reduction)

        x_hat = self.dec(y_hat)
        x_hat = x_hat * q_g
        output= self.refine(x_hat)
        return output
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None