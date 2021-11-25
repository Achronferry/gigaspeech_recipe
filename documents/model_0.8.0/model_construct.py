import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from .modules.mask_estimator import MaskEstimator
from .modules.beamformer import DNN_Beamformer
from espnet.nets.pytorch_backend.frontends.feature_transform import FeatureTransform
from pytorch_backend.models.modules.transducer import Transducer

def extract_from_args(args, ls):
    return {i: getattr(args, i) for i in ls}

def parse_model_args(module_list, config):

    me, bf, be, bd = module_list
    parser = argparse.ArgumentParser(description='Process model configures')
    frontend_group = parser.add_argument_group('frontend', 'frontend configures')

    backend_group = parser.add_argument_group('backend', 'backend configures')


    if me is not None and me.lower() == 'masknet':
        pass
    if bf is not None and bf.lower() == 'masknet':
        pass  

    backend_group.add_argument('--feat_size', default=None, type=int,
                        help='size of feature')
    backend_group.add_argument('--hidden_size', default=None, type=int,
                        help='hidden size of background input')
    backend_group.add_argument('--back_dropout', default=None, type=float,
                        help='hidden size of background input')   
    if be.lower() == "t":
        backend_group.add_argument('--eheads', default=None, type=int,
                            help='# of heads for background encoder transformer')
        backend_group.add_argument('--effdim', default=None, type=int,
                            help='feadforward dim for background encoder transformer')  
        backend_group.add_argument('--elayers', default=None, type=int,
                            help='# of layers for background encoder transformer')  
    else: 
        # backend_group.add_argument('--eunits', default=None, type=int,
        #                     help='# of units for background encoder rnn')
        backend_group.add_argument('--esubsample', default=None, type=str,
                            help='subsample rate for each layer in rnn')  
        backend_group.add_argument('--elayers', default=None, type=int,
                            help='# of layers for background encoder transformer')                                
    backend_group.add_argument('--num_vocabs', default=None, type=int,
                        help='Max. number of chars')
    backend_group.add_argument('--embed_dim', default=None, type=int,
                        help='Max. number of chars')
    
    if bd.lower() == "transducer":
        backend_group.add_argument('--dunits', default=None, type=int,
                            help='Max. number of chars')
        backend_group.add_argument('--dtype', default=None, type=str,
                            help='type of transducer predictor')  
        backend_group.add_argument('--dlayers', default=None, type=int,
                            help='Max. number of chars') 
        backend_group.add_argument('--djoint_dim', default=None, type=int,
                            help='Max. number of chars')


    args = parser.parse_args(config)
    return args


def construct_models(model_type, unparsed_model_args, hyper_save_path=None):
    
    module_list = model_type.split('-')

    use_front_end = False
    if len(module_list) == 2:
        module_list = [None, None] + module_list

    args = parse_model_args(module_list, unparsed_model_args)
    args.model_type = model_type
    
    me, bf, be, bd = module_list
    config_used = []

    if me is None:
        mask_estimator = None
    elif me.lower() == 'masknet':
        config_used += ['mtype', 'in_size', 'mlayers', 'munits', 'mprojs', 'front_dropout', 'mnmask']
        mask_estimator = MaskEstimator(args.mtype, args.in_size, args.mlayers, 
                                        args.munits, args.mprojs, args.front_dropout, nmask=args.mnmask + 1)
    else:
        raise NotImplementedError

    if bf is None:
        beamformer = None
        feature_trans = None
    elif use_front_end and bf.lower() == 'dnnmvdr':
        config_used += ['in_size', 'mnmask', 'bunits']
        beamformer = DNN_Beamformer(args.in_size, args.bunits, args.mnmask, btype='mvdr')
        feature_trans = FeatureTransform(n_fft=(args.in_size - 1) * 2) #TODO add config
    else:
        raise NotImplementedError

    front2back = nn.Sequential(nn.Linear(args.feat_size, args.hidden_size),
                                 nn.LayerNorm(args.hidden_size))
    if be.lower() == 't':
        config_used += ['hidden_size', 'eheads', 'effdim', 'back_dropout', 'elayers']
        encoder_layers = nn.TransformerEncoderLayer(args.hidden_size, args.eheads, args.effdim, args.back_dropout, batch_first=True)
        backend_enc = nn.TransformerEncoder(encoder_layers, args.elayers)
    else:
        from espnet.nets.pytorch_backend.transducer.rnn_encoder import Encoder
        config_used += ['hidden_size', 'back_dropout','esubsample' ,'elayers']
        subsumple = np.ones(args.elayers + 1, dtype=np.int)
        if args.esubsample is not None:
            for i,s in enumerate(args.esubsample.split('_')):
                subsumple[i] = int(s)
        backend_enc = Encoder(etype=be.lower(), idim=args.hidden_size, elayers=args.elayers, eunits=args.hidden_size, 
                             eprojs=args.hidden_size, subsample=subsumple, dropout=args.back_dropout)


    if bd.lower() == 'transducer':
        ['hidden_size', 'dunits', 'dtype', 'num_vocabs', 'dlayers', 'embed_dim', 'djoint_dim', 'back_dropout']
        backend_dec = Transducer(args.hidden_size, args.dunits, args.dtype, args.num_vocabs, args.dlayers, 
                                    args.embed_dim, args.djoint_dim, dropout=args.back_dropout)
    elif bd.lower() == 'ctc':
        raise NotImplementedError
    else:
        raise NotImplementedError


    model = E2E(mask_estimator, beamformer, feature_trans,front2back, backend_enc, backend_dec)
    if hyper_save_path is not None:
        with open(hyper_save_path, 'w') as f:
            save_conf = {'model_type': model_type}
            config_dict = extract_from_args(args, list(set(config_used)))
            save_conf.update(config_dict)
            yaml.dump(config_dict, f)
    return model




class E2E(nn.Module):
    def __init__(self, mask_estimator, beamformer, feature_trans, front2back, backend_enc, backend_dec):
        super().__init__()

        self.mask_estimator = mask_estimator
    
        self.beamformer = beamformer
        self.feature_transform = feature_trans

        self.front2back = front2back
        self.backend_enc = backend_enc
        self.backend_dec = backend_dec


    def forward(self, padded_input_spec, padded_token_ids, input_lens, 
                padded_time_anotation=None, spk_nums=None):
        '''
            padded_input_spec (B,C,T,F), 
            padded_time_anotation (B,J,T), 
            padded_token_ids (B,L),
            spk_nums (B,),
            input_lens (B,),
        '''
    
        bsize = padded_input_spec.shape[0]
        if len(padded_input_spec.shape) == 4:
            padded_input_spec = padded_input_spec[:, :, :max(input_lens), : ]
            masks, _ = self.mask_estimator(padded_input_spec, input_lens) # tuple of masks(N,B,C,T,F)
            enhanced, _, ws = self.beamformer(padded_input_spec, input_lens, masks) # (n, B, T, F)
            # assert self.nmask == len(masks)
            # TODO decide use which enhanced
            hs_pad = enhanced[0] #(B, T, D)
        else:
            padded_input_spec = padded_input_spec[:, :max(input_lens), : ]
            hs_pad = padded_input_spec

        src_padding_mask = torch.zeros(hs_pad.shape[:-1], device=hs_pad.device).bool()  # (B * T)
        for idx, l in enumerate(input_lens):
            src_padding_mask[idx, l:] = 1

        hs_pad = self.front2back(hs_pad)
        if type(self.backend_enc) == nn.TransformerEncoder:
            enc_out = self.backend_enc(hs_pad, src_key_padding_mask=src_padding_mask)
        else:
            enc_out, input_lens, hs_mask = self.backend_enc(hs_pad, input_lens)


        trans_loss, outputs = self.backend_dec(enc_out, input_lens, padded_token_ids)

        return trans_loss * bsize
    
    def recognize(self, input_spec, input_lens, time_anotation=None, **beam_kargs):
        '''
        input_spec: (B,C,T,F)
        '''

        if len(input_spec.shape) == 4:
            input_spec = input_spec[:, :, :max(input_lens), : ]
            masks, _ = self.mask_estimator(input_spec, input_lens) # tuple of masks(N,B,C,T,F)
            enhanced, _, ws = self.beamformer(input_spec, input_lens, masks) # (n, B, T, F)
            # assert self.nmask == len(masks)
            # TODO decide use which enhanced
            hs_pad = enhanced[0] #(B, T, D)
        else:
            input_spec = input_spec[:, :max(input_lens), : ]
            hs_pad = input_spec
        
        src_padding_mask = torch.zeros(hs_pad.shape[:-1], device=hs_pad.device).bool()  # (B * T)
        for idx, l in enumerate(input_lens):
            src_padding_mask[idx, l:] = 1

        hs_pad = self.front2back(hs_pad)
        if type(self.backend_enc) == nn.TransformerEncoder:
            enc_out = self.backend_enc(hs_pad, src_key_padding_mask=src_padding_mask)
        else:
            enc_out, input_lens, hs_mask = self.backend_enc(hs_pad, input_lens)

        pred = []
        for (i, ilen) in zip(enc_out, input_lens):
            pred.append(self.backend_dec.recognize(i[:ilen], **beam_kargs))
        return pred

        
