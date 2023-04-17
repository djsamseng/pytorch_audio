#!/usr/bin/env python3
import logging
import pathlib
from argparse import ArgumentParser

import typing

import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchaudio
import torchaudio.models.decoder
import torchaudio.utils
import torchaudio.datasets

from common import MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_MUSTC, MODEL_TYPE_TEDLIUM3
from librispeech.lightning import LibriSpeechRNNTModule
from mustc.lightning import MuSTCRNNTModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tedlium3.lightning import TEDLIUM3RNNTModule

torch.set_float32_matmul_precision(precision="medium")

def get_trainer(args):
    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
        verbose=True,
    )
    callbacks = [
        checkpoint,
        train_checkpoint,
    ]
    return Trainer(
        #default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        #num_nodes=args.num_nodes,
        #gpus=args.gpus,
        accelerator="gpu",
        #strategy="ddp",
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        precision="32",
        limit_train_batches=100,
        limit_val_batches=100,
        limit_test_batches=1
    )


def get_lightning_module(args):
    if args.model_type == MODEL_TYPE_LIBRISPEECH:
        return LibriSpeechRNNTModule(
            librispeech_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    elif args.model_type == MODEL_TYPE_TEDLIUM3:
        return TEDLIUM3RNNTModule(
            tedlium_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    elif args.model_type == MODEL_TYPE_MUSTC:
        return MuSTCRNNTModule(
            mustc_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    else:
        raise ValueError(f"Encountered unsupported model type {args.model_type}.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-type", type=str, choices=[MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_TEDLIUM3, MODEL_TYPE_MUSTC], required=True
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
        required=True,
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        help="Path to datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--num-nodes",
        default=4,
        type=int,
        help="Number of nodes to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--gpus",
        default=8,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 8)",
    )
    parser.add_argument(
        "--epochs",
        default=120,
        type=int,
        help="Number of epochs to train for. (Default: 120)",
    )
    parser.add_argument(
        "--gradient-clip-val", default=10.0, type=float, help="Value to clip gradient values to. (Default: 10.0)"
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")

LJSPEECHITEM = typing.Tuple[torch.Tensor, int, str, str]
class WaveformTransformPipeline(torch.nn.Module):
  def __init__(self, input_freq: int, output_freq: int, n_fft: int, n_mels: int, stretch_factor: float) -> None:
    super().__init__()
    self.resample = torchaudio.transforms.Resample(orig_freq=input_freq, new_freq=output_freq)
    self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=output_freq, n_fft=n_fft, n_mels=n_mels, hop_length=160)

  def forward(self, args: torch.Tensor) -> torch.Tensor: # type: ignore
    resampled = self.resample(args)
    mel = self.melspec(resampled)
    mel = torch.nn.utils.rnn.pad_sequence(mel, batch_first=True)
    mel = 10 * torch.log10(mel)
    if False:
      plt.figure()
      plt.pcolormesh(mel[0])
      plt.figure()
      plt.specgram(resampled[0], Fs=16000, scale="dB")
      plt.show()
    return mel

def train_collate_fn(BD: typing.List[LJSPEECHITEM]):
  pipeline = WaveformTransformPipeline(input_freq=22050, output_freq=16000, n_fft=400, n_mels=80, stretch_factor=1.0)
  features = [pipeline(D[0]) for D in BD]
  targets = [D[2] for D in BD]
  return features, targets

def get_dataset(
  batch_size: int = 1
):
  ljspeech = torchaudio.datasets.LJSPEECH(root="/home/samuel/dev/tess/audio/datasets/")
  train_size = int(0.8 * len(ljspeech))
  test_size = len(ljspeech) - train_size
  train_ds, test_ds = typing.cast(
    typing.Tuple[torchaudio.datasets.LJSPEECH, torchaudio.datasets.LJSPEECH],
    torch.utils.data.random_split(dataset=ljspeech, lengths=[train_size, test_size])
  )
  train_dl = typing.cast(
    torch.utils.data.DataLoader[torchaudio.datasets.LJSPEECH],
    torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)
  )
  test_dl = typing.cast(
    torch.utils.data.DataLoader[torchaudio.datasets.LJSPEECH],
    torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)
  )
  return train_dl, test_dl,

def cli_main():
    args = parse_args()
    init_logger(args.debug)
    model = get_lightning_module(args)
    trainer = get_trainer(args)
    trainer.fit(model)
    print("Fit!")


# sgp python3 train.py --model-type librispeech --dataset-path ./datasets/ --sp-model-path ./librispeech/spm_bpe_4096_librispeech.model --gpus 1 --epochs 1 --debug --global-stats-path ./global_stats.json
# sgp python3 -m tensorboard.main --logdir ./lightning_logs/
if __name__ == "__main__":
    cli_main()
