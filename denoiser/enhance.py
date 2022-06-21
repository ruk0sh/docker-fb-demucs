# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
import sys

import torch
import torchaudio

from .audio import Audioset, find_audio_files
from . import distrib, pretrained
from .demucs import DemucsStreamer

from .utils import LogProgress

logger = logging.getLogger(__name__)


def add_flags(parser):
    """
    Add the flags for the argument parser that are related to model loading and evaluation"
    """
    pretrained.add_model_flags(parser)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--dry', type=float, default=0,
                        help='dry/wet knob coefficient. 0 is only denoised, 1 only input signal.')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--streaming', action="store_true",
                        help="true streaming evaluation for Demucs")


parser = argparse.ArgumentParser(
        'denoiser.enhance',
        description="Speech enhancement using Demucs - Generate enhanced files")
add_flags(parser)
parser.add_argument("--out_dir", type=str, default="enhanced",
                    help="directory putting enhanced wav files")
parser.add_argument("--converted_dir", type=str, default="converted",
                    help="directory putting converted wav files")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("-v", "--verbose", action="store_const", const=logging.DEBUG,
                    default=logging.INFO, help="more loggging")
parser.add_argument("-e","--valid_extensions", nargs="+", required=False, default=["wav"],
                    help="Provide valid audioextensions, space separated.")
parser.add_argument("--sr", required=False, type=int, default=None,
                    help='Sample rate for output audio files.')
parser.add_argument("--keep_noisy", required=False, action="store_true",
                    help="If specified, original noisy audios will be saved to output folder.")

group = parser.add_mutually_exclusive_group()
group.add_argument("--noisy_dir", type=str, default=None,
                   help="directory including noisy wav files")
group.add_argument("--noisy_json", type=str, default=None,
                   help="json file including noisy wav files")


def get_estimate(model, noisy, args):
    torch.set_num_threads(1)
    if args.streaming:
        streamer = DemucsStreamer(model, dry=args.dry)
        with torch.no_grad():
            estimate = torch.cat([
                streamer.feed(noisy[0]),
                streamer.flush()], dim=1)[None]
    else:
        with torch.no_grad():
            estimate = model(noisy)
            estimate = (1 - args.dry) * estimate + args.dry * noisy
    return estimate


def save_wavs(estimates, noisy_sigs, filenames, out_dir, resampler, sr=16_000, keep_noisy=False):
    # Write result
    for estimate, noisy, filename in zip(estimates, noisy_sigs, filenames):
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        if keep_noisy:
            write(noisy, filename + "_noisy.wav", resampler, sr=sr)
        write(estimate, filename + "_enhanced.wav", resampler, sr=sr)


def write(wav, filename, resampler, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    if resampler:
        wav = resampler(wav)
    torchaudio.save(filename, wav.cpu(), sr)


def get_dataset(args, sample_rate, channels):
    if hasattr(args, 'dset'):
        paths = args.dset
    else:
        paths = args
    if paths.noisy_json:
        with open(paths.noisy_json) as f:
            files = json.load(f)
    elif paths.noisy_dir:
        exts = [f".{ext}" for ext in args.valid_extensions]
        files = find_audio_files(paths.noisy_dir, converted_dir=args.converted_dir, exts=exts)
    else:
        logger.warning(
            "Small sample set was not provided by either noisy_dir or noisy_json. "
            "Skipping enhancement.")
        return None
    return Audioset(files, with_path=True,
                    sample_rate=sample_rate, channels=channels, convert=True)


def _estimate_and_save(model, noisy_signals, filenames, out_dir, args, resampler):
    estimate = get_estimate(model, noisy_signals, args)
    sr = args.sr if args.sr else model.sample_rate
    save_wavs(estimate, noisy_signals, filenames, out_dir, resampler, sr=sr, keep_noisy=args.keep_noisy)


def enhance(args, model=None, local_out_dir=None):
    # Load model
    if not model:
        model = pretrained.get_model(args).to(args.device)
    model.eval()
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir

    dset = get_dataset(args, model.sample_rate, model.chin)
    if dset is None:
        return
    loader = distrib.loader(dset, batch_size=1)

    resampler = None
    if args.sr and args.sr != model.sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=model.sample_rate,
            new_freq=args.sr,
            resampling_method="sinc_interpolation",
        )

    if distrib.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    distrib.barrier()

    with ProcessPoolExecutor(args.num_workers) as pool:
        iterator = LogProgress(logger, loader, name="Generate enhanced files")
        pendings = []
        for data in iterator:
            # Get batch data
            noisy_signals, filenames = data
            noisy_signals = noisy_signals.to(args.device)
            if args.device == 'cpu' and args.num_workers > 1:
                pendings.append(
                    pool.submit(_estimate_and_save,
                                model, noisy_signals, filenames, out_dir, args, resampler))
            else:
                # Forward
                estimate = get_estimate(model, noisy_signals, args)
                sr = args.sr if args.sr else model.sample_rate
                save_wavs(estimate, noisy_signals, filenames, out_dir, resampler, sr=sr)

        if pendings:
            print('Waiting for pending jobs...')
            for pending in LogProgress(logger, pendings, updates=5, name="Generate enhanced files"):
                pending.result()


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    enhance(args, local_out_dir=args.out_dir)
