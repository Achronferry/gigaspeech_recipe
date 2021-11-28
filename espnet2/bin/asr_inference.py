#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List

from espnet2.asr.beam_search.beam_search_transducer import BeamSearchTransducer  # espnet2
from espnet2.nets.transformer.subsampling import TooShortUttError
from espnet2.utils.cli_utils import get_commandline_args
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.train.build_task import ASRTask
# from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
from espnet2.asr.beam_search.scorer_interface import Hypothesis


class Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        dtype: str = "float32",
        beam_size: int = 20,
        nbest: int = 1,
        beam_search_conf: dict = {}, # more args found in beamsearchtransducer
        # TODO
        # lm_weight: float = 1.0,
        # lm_train_config: Union[Path, str] = None,
        # lm_file: Union[Path, str] = None,
        streaming: bool = False,
        # batch_size: int = 1,
    ):
        assert not streaming
        assert check_argument_types()

        # 1. Build ASR model
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        token_list = asr_model.token_list

        # 2. Build Language model
        # if lm_train_config is not None:
        #     lm, lm_train_args = LMTask.build_model_from_file(
        #         lm_train_config, lm_file, device
        #     )
        #     scorers["lm"] = lm.lm

        # 3. Build BeamSearch object
        beam_search = BeamSearchTransducer(
            decoder=asr_model.decoder,
            joint_network=asr_model.joint_network,
            beam_size=beam_size,
            lm=None,
            lm_weight=0,
            nbest=nbest,
            **beam_search_conf
        ).to(device=device, dtype=getattr(torch, dtype)).eval()

        logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

    @torch.no_grad()
    def __call__(
        self, batch: Dict[str, torch.Tensor]
    ):
        """Inference

        Args:
            data: Input speech data
        Returns:
            batchfied, nbest, tuple of [text, token, token_int, hyp]

        """
        assert check_argument_types()

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        encoder_out, encoder_out_lens = self.asr_model.encode(**batch)

        # c. Passed the encoder result and the beam search
        batch_nbest_hyps = self.beam_search(encoder_out, encoder_out_lens)

        results = []
        for nbest_hyps in batch_nbest_hyps:
            result = []
            for hyp in nbest_hyps[: self.nbest]:
                assert isinstance(hyp, Hypothesis), type(hyp)

                # remove sos/eos and get results
                token_int = hyp.yseq[1:-1].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(filter(lambda x: x != 0, token_int))

                # Change integer-ids to tokens
                token = self.converter.ids2tokens(token_int)

                if self.tokenizer is not None:
                    text = self.tokenizer.tokens2text(token)
                else:
                    text = None
                result.append((text, token, token_int, hyp))
            results.append(result)

        return results


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: str,
    asr_model_file: str,
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    streaming: bool,
    
    # lm_train_config: Optional[str],
    # lm_file: Optional[str],
    # word_lm_train_config: Optional[str],
    # word_lm_file: Optional[str],
    # lm_weight: float,
    # penalty: float,
    **kwargs,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    from espnet2.utils import setup_logging_config
    setup_logging_config(log_level)

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text = Speech2Text(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        dtype=dtype,
        beam_size=beam_size,
        nbest=nbest,
        streaming=streaming,
    )

    # 3. Build data-iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # kwargs['resume'] : do not inference if done
    if kwargs['resume']:
        import subprocess, os
        output_file = f'{output_dir}/{nbest}best_recog/text'
        logging.warning(output_file)
        if os.path.exists(output_file):
            cnt = subprocess.run(['wc', '-l', key_file], stdout=subprocess.PIPE).stdout.decode('utf-8').split(' ')[0]
            cnt2 = subprocess.run(['wc', '-l', output_file], stdout=subprocess.PIPE).stdout.decode('utf-8').split(' ')[0]
            if cnt == cnt2:
                logging.warning("Already inferred, do not do this again!")
                return

    # 6.5 Load reference text file
    ref_text = {}
    if kwargs['reference_file'] is not None:
        with open(kwargs['reference_file'], 'r') as f:
            for line in f.readlines():
                import re
                data = re.split(' |\t', line.strip(' \n'))
                ref_text[data[0]] = data[1:]

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            logging.debug(f"batch({len(keys)}) Decode from {keys}")

            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"keys len:{len(keys)} != batch_size:{_bs}"
            # batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # results [batch_size, Nbest_list(N, tuple)]
            # N-best list of (text, token, token_int, hyp_object)
            try:
                import time
                if kwargs["report_rtf"]:
                    t = time.perf_counter()
                results = speech2text(batch)
                if kwargs["report_rtf"]:
                    rtf = (time.perf_counter() - t) / sum(batch["speech_lengths"]).item() # rtf = forward time / speech wav length
            except TooShortUttError as e:
                logging.error(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [(" ", ["<space>"], [2], hyp) * nbest] * _bs

            for idx, result in enumerate(results):
                key = keys[idx]
                if kwargs["report_rtf"]:
                    assert len(keys) == 1, "rtf require batch_size=1,single core"
                    writer["rtf"][key] = str(rtf)

                report_detail=True
                report(speech2text, writer, key, result, ref_text=ref_text, report_detail=report_detail)

                # writing
                for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), result):
                    # Create a directory: outdir/{n}best_recog
                    ibest_writer = writer[f"{n}best_recog"]

                    # Write the result to each file
                    ibest_writer["token"][key] = " ".join(token)
                    ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["score"][key] = str(hyp.score)

                    if text is not None:
                        ibest_writer["text"][key] = text
                    else:
                        logging.warning(f"{key} does not have text")



def report(speech2text, writer, key, result, ref_text=None, report_detail=True):
    """Merged report
        - beamsearch: d. report the best result borrowed from espnet2/nets/BeamSearch
        - inference: d.1 report the best result

    result: N-best list of (text, token, token_int, hyp_object)
    """
    if report_detail:
        best_hyp = result[0][-1]

        logging.info(f"total log probability: {best_hyp.score:.2f}")
        logging.info(f"normalized log probability: {best_hyp.score / len(best_hyp.yseq):.2f}")
        logging.info(f"total number of ended hypotheses: {len(result)}")

    error_writer = writer['err_uttid'] # only write to best
    error_writer.check_id = False # write to err_uttid
    
    if speech2text.asr_model.token_list is not None:
        logging.info(f"Result on {key}")
        # _yseq = [speech2text.asr_model.token_list[x] for x in results[idx][-1].yseq[1:-1]]
        # _yseq = "".join([_y if _y != "<space>" else " " for _y in _yseq])
        for nbest_idx, res_tuple in enumerate(result):
            log = f"\n\thyp({nbest_idx}): " + res_tuple[0]
            if ref_text:
                _ref = " ".join(ref_text[key])
                log += "\n\tref   : " + _ref

            if ref_text and _ref != res_tuple[0]:
                logging.warning(log)
                if nbest_idx == 0:
                    error_writer[key] = key
            else:
                logging.info(log)



def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )
    parser.add_argument("--resume", type=str2bool, default=True, help="do not run finished tasks again.")
    parser.add_argument("--report_rtf", type=str2bool, default=False, help="Report RTF")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--reference_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)
    group.add_argument("--lm_train_config", type=str)
    group.add_argument("--lm_file", type=str)
    group.add_argument("--word_lm_train_config", type=str)
    group.add_argument("--word_lm_file", type=str)

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")

    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--streaming", type=str2bool, default=False)

    group.add_argument("--beam_search_conf", type=dict, default={}, help="transducer conf for beamsearchtransducer")


    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
