#!/bin/bash

set -e

umask o+w

exec python -m denoiser.enhance "$@"
