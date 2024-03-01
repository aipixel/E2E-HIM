import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Module, Sequential, Conv2d, Parameter,Softmax
from positionembedding import PositionEmbeddingSine
from einops import rearrange,repeat,reduce
import swin
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos ):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask  = None,
                     memory_mask  = None,
                     tgt_key_padding_mask  = None,
                     memory_key_padding_mask  = None,
                     pos  = None,
                     query_pos  = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask  = None,
                    memory_mask  = None,
                    tgt_key_padding_mask  = None,
                    memory_key_padding_mask  = None,
                    pos  = None,
                    query_pos  = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask  = None,
                memory_mask  = None,
                tgt_key_padding_mask  = None,
                memory_key_padding_mask  = None,
                pos  = None,
                query_pos  = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class PAM_Module(nn.Module):
    def __init__(self, in_dim,inter_channels):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=inter_channels, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=inter_channels, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask  = None,
                memory_mask  = None,
                tgt_key_padding_mask  = None,
                memory_key_padding_mask  = None,
                pos  = None,
                query_pos  = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class GNs_(nn.Module):
    def __init__(self,ch):
        super(GNs_, self).__init__()
        if ch%32==0:
            g=ch//32
        else:
            g=ch//16
        self.gn=nn.GroupNorm(g,ch)
    def forward(self,x):
        return self.gn(x)


class RESBASEE(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(RESBASEE, self).__init__()
        self.in_conv1 =  nn.Sequential(nn.Conv2d(inplanes, planes, 1,1,0),GNs_(planes),nn.ReLU(inplace=True),conv3x3(planes, planes, stride),GNs_(planes),nn.LeakyReLU(inplace=True),nn.Conv2d(planes, inplanes,1,1,0),GNs_(inplanes))
        self.in_prelu1 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.in_conv1(x)+x
        out = self.in_prelu1(out)
        return out


class E2EHIMST(nn.Module):
    def __init__(self):
        super(E2EHIMST, self).__init__()
        self.pre_conv = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=True),nn.GroupNorm(16, 32),nn.ReLU(True),nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),nn.GroupNorm(16, 32),nn.ReLU(True))
        self.res1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),nn.GroupNorm(32,64),nn.ReLU())
        self.ds1 = nn.Sequential((nn.Conv2d(64,128,3,2,1,bias=True)),nn.GroupNorm(32,128),nn.ReLU())
        trans = swin.SwinTransformer(pretrain_img_size=224,
                                     embed_dim=96,
                                     depths=[2, 2, 6, 2],
                                     num_heads=[3, 6, 12, 24],
                                     window_size=7,
                                     ape=False,
                                     drop_path_rate=0.2,
                                     patch_norm=True,
                                     use_checkpoint=False)
        trans.patch_embed.proj = nn.Conv2d(128, 96, kernel_size=1, stride=1)
        self.trans=  trans
        self.maxp=20
        self.toplayer = nn.Sequential(nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0),nn.GroupNorm(32,256),nn.ReLU(True))
        decoder1 = TransformerDecoderLayer(256,8, 2048,0.1)
        norm=nn.Sequential(nn.LayerNorm(256))
        self.decoder1 = TransformerDecoder(decoder1, 6,norm)
        self.query_embed = nn.Embedding(self.maxp, 256)
        self.globalfcnum=MLP(256,128,2,2)
        self.globalfc=MLP(258,512,(256+1)*3,3)
        self.globalfc2=MLP(258,512,256*3+3+64+1,3)
        self.globalfck=MLP(258,512,128*2,1)
        self.conv1=nn.Conv2d(256,1,1,1,0,bias=True)
        self.conv2=nn.Conv2d(256,1,1,1,0,bias=True)
        self.c32m=nn.Sequential(nn.Conv2d(768,512,1,1,0,bias=True),nn.GroupNorm(16, 512), nn.ReLU(inplace=True))
        self.c16m=nn.Sequential(nn.Conv2d(896,512,3,1,1,bias=True),nn.GroupNorm(16, 512), nn.ReLU(inplace=True),RESBASEE(512,128),RESBASEE(512,128))
        self.c8m=nn.Sequential(nn.Conv2d(704,256,3,1,1,bias=True),nn.GroupNorm(16, 256), nn.ReLU(inplace=True),RESBASEE(256,64),RESBASEE(256,64),nn.Conv2d(256,256,1,1,0,bias=True),nn.GroupNorm(16, 256), nn.ReLU(inplace=True))
        self.c8d=nn.Sequential(nn.Conv2d(512+self.maxp,512,1,1,0,bias=True),nn.GroupNorm(32,512),nn.ReLU(inplace=True),RESBASEE(512,128),RESBASEE(512,128))
        self.c8d2=nn.Sequential(nn.Conv2d(512,512,3,2,1,bias=True),nn.GroupNorm(32, 512), nn.ReLU(inplace=True))
        self.c8d3 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.GroupNorm(32, 512), nn.ReLU(inplace=True),
                                  nn.Dropout(0.1), nn.Conv2d(512, 256, 3, 1, 1, bias=True), nn.GroupNorm(32, 256),
                                  nn.ReLU(inplace=True))
        self.c16 = nn.Sequential(nn.Conv2d(1408,512,1,1,0,bias=True),nn.GroupNorm(32,512),nn.LeakyReLU(inplace=True),RESBASEE(512,128),RESBASEE(512,128))
        self.c16d = nn.Sequential(nn.Conv2d(512,768,3,2,1,bias=True),nn.GroupNorm(32, 768), nn.LeakyReLU(inplace=True))
        self.c32 = nn.Sequential(nn.Conv2d(512+1024,1024,1,1,0,bias=True),nn.GroupNorm(32,1024),nn.LeakyReLU(inplace=True),RESBASEE(1024,256))
        self.c32b = nn.Sequential(nn.Conv2d(1024,1024,1,1,0,bias=True),nn.GroupNorm(32,1024),nn.LeakyReLU(inplace=True),RESBASEE(1024,256),RESBASEE(1024,256))
        self.pam=PAM_Module(1024,256)
        self.c16u = nn.Sequential( nn.Conv2d(1920,512,1,1,0,bias=True),nn.GroupNorm(32, 512), nn.LeakyReLU(inplace=True),RESBASEE(512,128),RESBASEE(512,128))
        self.c8=nn.Sequential(nn.Conv2d(1216,512,1,1,0,bias=True),nn.GroupNorm(32,512),nn.LeakyReLU(inplace=True),RESBASEE(512,128),RESBASEE(512,128))
        self.c4=nn.Sequential(nn.Conv2d(736,256,3,1,1,bias=True),nn.GroupNorm(32,256),nn.LeakyReLU(inplace=True),RESBASEE(256,64),RESBASEE(256,64),RESBASEE(256,64))
        self.c2=nn.Sequential(nn.Conv2d(256+64,128,3,1,1,bias=True),nn.GroupNorm(32,128),nn.LeakyReLU(inplace=True),nn.Conv2d(128,128,3,1,1,bias=True),nn.GroupNorm(32,128),nn.LeakyReLU(inplace=True))
        self.c1=nn.Sequential(nn.Conv2d(128+3,96,3,1,1,bias=True),nn.LeakyReLU(True),nn.Conv2d(96,64,3,1,1,bias=True),nn.LeakyReLU(True))
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.up8x=nn.Upsample(scale_factor=8,mode='bilinear',align_corners=False)
        self.c1t1=nn.Conv2d(256,1,1,1,0,bias=True)
        self.c1t2=nn.Conv2d(256,1,1,1,0,bias=True)
        self.drop=nn.Dropout(0.0)
        self.down=nn.Upsample(scale_factor=0.5,mode='bilinear',align_corners=False)
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.up8x=nn.Upsample(scale_factor=8,mode='bilinear',align_corners=False)
        self.up8xb=nn.Upsample(scale_factor=8,mode='nearest')
        self.up16x=nn.Upsample(scale_factor=16,mode='bilinear',align_corners=False)
        self.pe=PositionEmbeddingSine(128)
        self.otherwright1=nn.Sequential(nn.Conv2d(256,128,1,1,0))
        self.otherwright2=nn.Sequential(nn.Conv2d(256,128,1,1,0))
        self.g1=nn.GroupNorm(32,128)
        self.g2=nn.GroupNorm(32,128)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        img = x
        bs,ch,imgh,imgw=img.shape
        x2a=self.pre_conv(img)
        x2b=self.res1(x2a)
        x4a=self.ds1(x2b)
        x4,x8,x16,x32=self.trans(x4a)
        feats=self.toplayer(x32)
        featsm=rearrange(feats,"b c h w->(h w ) b c")
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1,bs,1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder1(tgt, featsm, query_pos=query_embed)
        persel=rearrange(hs,"t b c->b t c")
        x4o=self.globalfcnum(persel)
        x4os=torch.softmax(x4o,2)
        persel=torch.cat([persel,x4os],-1)
        tempkernel1=self.globalfc(persel)
        tempkernel1a,tempkernel1c,tempkernel1b,tempkernel1d,tempkernel1e,tempkernel1f= tempkernel1[:,:,0:256],tempkernel1[:,:,256:512] ,tempkernel1[:,:,512:513],tempkernel1[:,:,513:514],tempkernel1[:,:,514:514+256],tempkernel1[:,:,514+256:]
        tempkernel1b=tempkernel1b.view(self.maxp,1)
        tempkernel1a=rearrange(tempkernel1a,'b t (c1 c2 h w)->(b t) c1 c2 h w',c1=1,h=1,w=1)
        tempkernel1d=tempkernel1d.view(self.maxp,1)
        tempkernel1c=rearrange(tempkernel1c,'b t (c1 c2 h w)->(b t) c1 c2 h w',c1=1,h=1,w=1)
        tempkernel1f=tempkernel1f.view(self.maxp,1)
        tempkernel1e=rearrange(tempkernel1e,'b t (c1 c2 h w)->(b t) c1 c2 h w',c1=1,h=1,w=1)
        feats2=self.c32m(x32)
        feats2=self.up(feats2)
        feats2=self.c16m(torch.cat([feats2,x16],1))
        feats2up=torch.cat([self.up(feats2),x8],1)
        feats2up=self.c8m(feats2up)
        tc1=[]
        tc2=[]
        tc3=[]
        for cidx in range(self.maxp):
            tc1.append(F.conv2d(feats2up, tempkernel1a[cidx],tempkernel1b[cidx]))
            tc2.append(F.conv2d(feats2up, tempkernel1c[cidx],tempkernel1d[cidx]))
            tc3.append(F.conv2d(feats2up, tempkernel1e[cidx],tempkernel1f[cidx]))
        tc3=torch.cat(tc3,1)
        x4sel_=tc3
        x4sel =self.up8x(x4sel_)
        tc1_=tc1+[self.c1t1(feats2up)]
        tc2_=tc2+[self.c1t2(feats2up)]
        tc1_=torch.cat(tc1_,0)
        tc2_=torch.cat(tc2_,0)
        tc1_=torch.softmax(tc1_,0)
        tc2_=torch.softmax(tc2_,0)
        t2a_=  self.g1(self.otherwright1(feats2up))
        t2b_=  self.g2(self.otherwright2(feats2up))
        tempkernel2=self.globalfck(persel)
        tempkernel2a,tempkernel2b=tempkernel2[:,:,0:128],tempkernel2[:,:,128:]
        tempkernel2a=rearrange(tempkernel2a,'b t (c h w)->(b t) c h w',h=1,w=1)
        tempkernel2a=self.g1(repeat(tempkernel2a,'b c h w->b c (h h1) (w w1)',h1=80,w1=80))
        tempkernel2b=rearrange(tempkernel2b,'b t (c h w)->(b t) c h w',h=1,w=1)
        tempkernel2b=self.g2(repeat(tempkernel2b,'b c h w->b c (h h1) (w w1)',h1=80,w1=80))
        tempkernel2a=torch.cat([tempkernel2a,t2a_],0)
        tempkernel2b=torch.cat([tempkernel2b,t2b_],0)
        tc1=[]
        tc2=[]
        for cidx in range(self.maxp+1):
            tc1.append(tc1_[cidx]*tempkernel2a[cidx:cidx + 1] )
            tc2.append(tc2_[cidx]*tempkernel2b[cidx:cidx + 1] )
        tc1=sum(tc1)
        tc2=sum(tc2)
        segs=torch.sigmoid(x4sel_)
        masksum2=torch.cat([tc1,tc2,feats2up,segs],1)
        masksum16=self.c8d(masksum2)
        masksum16f=self.c8d2(masksum16)
        masksum16_=torch.cat([masksum16f,feats2,x16],1)
        masksum16_=self.c16(masksum16_)
        masksum16_d=self.c16d(masksum16_)
        masksum32_=torch.cat([masksum16_d,x32],1)
        masksum32_=self.c32(masksum32_)
        ppm_out=self.pam(masksum32_)
        ppm_out=self.c32b(ppm_out)
        masksum16_=torch.cat([masksum16_,self.up(ppm_out),x16],1)
        masksum16_=self.c16u(masksum16_)
        masksum16_=self.up(masksum16_)
        masksum8_=torch.cat([masksum16_,x8,masksum16],1)
        tc8=self.c8(masksum8_)
        masksum16up=self.c8d3(tc8)
        tempkernelo=self.globalfc2(persel)
        tempkernelo1,tempkernelo2,tempkernelo3,tempkernelo4=tempkernelo[:,:,0:256*3],tempkernelo[:,:,256*3:256*3+3],tempkernelo[:,:,256*3+3:256*3+3+64],tempkernelo[:,:,-1:]
        tempkernelo2=tempkernelo2.view(self.maxp,3)
        tempkernelo1=rearrange(tempkernelo1,'b t (c1 c2 h w)->(b t) c1 c2 h w',c1=3,h=1,w=1)
        tempkernelo4=tempkernelo4.view(self.maxp,1)
        tempkernelo3=rearrange(tempkernelo3,'b t (c1 c2 h w)->(b t) c1 c2 h w',c1=1,h=1,w=1)
        tc3 = []
        for cidx in range(self.maxp):
            tc3.append(F.conv2d(masksum16up, tempkernelo1[cidx], tempkernelo2[cidx]))
        tc3=torch.cat(tc3,1)
        masksum16up=self.up8x(tc3)
        masksum8_=self.up(torch.cat([tc8],1) )
        masksum4_=torch.cat([masksum8_,x4,x4a],1)
        masksum4_=self.up(self.c4(masksum4_))
        masksum2_=torch.cat([masksum4_,x2b],1)
        masksum2_=self.up(self.c2(masksum2_))
        masksum1_=torch.cat([masksum2_,x],1)
        x1all =self.c1(masksum1_)
        tc3 = []
        for cidx in range(self.maxp):
            tc3.append(F.conv2d(x1all, tempkernelo3[cidx], tempkernelo4[cidx]))
        tc3=torch.cat(tc3,1)
        x1all=torch.clamp(tc3,0,1)
        return x4o,x1all,masksum16up