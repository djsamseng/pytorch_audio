#!/usr/bin/env python3
"""The demo script for testing the pre-trained Emformer RNNT pipelines.

Example:
python pipeline_demo.py --model-type librispeech --dataset-path ./datasets/librispeech
"""
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt

import torch
import torchaudio
from common import MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_MUSTC, MODEL_TYPE_TEDLIUM3
from mustc.dataset import MUSTC
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH, RNNTBundle
#from torchaudio.prototype.pipelines import EMFORMER_RNNT_BASE_MUSTC, EMFORMER_RNNT_BASE_TEDLIUM3

logger = logging.getLogger(__name__)


@dataclass
class Config:
    dataset: Callable
    bundle: RNNTBundle


_CONFIGS = {
    MODEL_TYPE_LIBRISPEECH: Config(
        partial(torchaudio.datasets.LIBRISPEECH, url="test-clean"),
        EMFORMER_RNNT_BASE_LIBRISPEECH,
    ),
    # MODEL_TYPE_MUSTC: Config(
    #     partial(MUSTC, subset="tst-COMMON"),
    #     EMFORMER_RNNT_BASE_MUSTC,
    # ),
    # MODEL_TYPE_TEDLIUM3: Config(
    #     partial(torchaudio.datasets.TEDLIUM, release="release3", subset="test"),
    #     EMFORMER_RNNT_BASE_TEDLIUM3,
    # ),
}

def get_model_setup():
  num_symbols = 4097 # Librispeech
  # https://github.com/pytorch/audio/blob/9912e54df0522915f0f15dd637636327301b9d75/torchaudio/models/rnnt.py#L794
  model = torchaudio.models.emformer_rnnt_model(
    input_dim=80,
    encoding_dim=1024,
    num_symbols=num_symbols,
    segment_length=16,
    right_context_length=4,
    time_reduction_input_dim=128,
    time_reduction_stride=4,
    transformer_num_heads=8,
    transformer_ffn_dim=2048,
    transformer_num_layers=20,
    transformer_dropout=0.1,
    transformer_activation="gelu",
    transformer_left_context_length=30,
    transformer_max_memory_size=0,
    transformer_weight_init_scale_strategy="depthwise",
    transformer_tanh_on_mem=True,
    symbol_embedding_dim=512,
    num_lstm_layers=3,
    lstm_layer_norm=True,
    lstm_layer_norm_epsilon=1e-3,
    lstm_dropout=0.3,
  )
  return model

def run_eval_streaming(args):
    #dataset = _CONFIGS[args.model_type].dataset(args.dataset_path)
    ljspeech = torchaudio.datasets.LJSPEECH(root="/home/samuel/dev/tess/audio/datasets/")
    dataset = ljspeech
    bundle = _CONFIGS[args.model_type].bundle
    decoder = bundle.get_decoder()
    model = get_model_setup()
    decoder = torchaudio.models.RNNTBeamSearch(model=model, blank=4096)
    token_processor = bundle.get_token_processor()
    feature_extractor = bundle.get_feature_extractor()
    streaming_feature_extractor = bundle.get_streaming_feature_extractor()
    #print(decoder, token_processor, feature_extractor, streaming_feature_extractor)
    # <torchaudio.pipelines.rnnt_pipeline._SentencePieceTokenProcessor object at 0x7fabed31af10> _ModuleFeatureExtractor(
    #   (pipeline): Sequential(
    #     (0): MelSpectrogram(
    #       (spectrogram): Spectrogram()
    #       (mel_scale): MelScale()
    #     )
    #     (1): _FunctionalModule()
    #     (2): _FunctionalModule()
    #     (3): _GlobalStatsNormalization()
    #     (4): _FunctionalModule()
    #   )
    # ) _ModuleFeatureExtractor(
    #   (pipeline): Sequential(
    #     (0): MelSpectrogram(
    #       (spectrogram): Spectrogram()
    #       (mel_scale): MelScale()
    #     )
    #     (1): _FunctionalModule()
    #     (2): _FunctionalModule()
    #     (3): _GlobalStatsNormalization()
    #   )
    # )
    hop_length = bundle.hop_length
    num_samples_segment = bundle.segment_length * hop_length
    num_samples_segment_right_context = num_samples_segment + bundle.right_context_length * hop_length

    for idx in range(1):
        sample = dataset[idx]
        waveform = sample[0].squeeze()
        # Streaming decode.
        state, hypothesis = None, None
        for idx in range(0, len(waveform), num_samples_segment):
            segment = waveform[idx : idx + num_samples_segment_right_context]
            segment = torch.nn.functional.pad(segment, (0, num_samples_segment_right_context - len(segment)))
            with torch.no_grad():
                features, length = streaming_feature_extractor(segment)
                plt.pcolormesh(features.rot90())
                plt.show()
                return
                # features = [t, n_mels] = [<=21, 80]
                hypos, state = decoder.infer(features, length, 10, state=state, hypothesis=hypothesis)
            hypothesis = hypos[0]
            transcript = token_processor(hypothesis[0], lstrip=False)
            print(transcript, end="", flush=True)

        # Non-streaming decode.
        with torch.no_grad():
            features, length = feature_extractor(waveform)
            hypos = decoder(features, length, 10)
        print("\nFinal non streaming:", token_processor(hypos[0][0]))
        print()


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--model-type", type=str, choices=_CONFIGS.keys(), required=True)
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        help="Path to dataset.",
        required=True,
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    run_eval_streaming(args)


if __name__ == "__main__":
    cli_main()
