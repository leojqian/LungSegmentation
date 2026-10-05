# Finding the U-Net checkpoint, and fetching it on first use.
#
# The checkpoint (~126MB) is too big for git, so it ships as a GitHub release
# asset and is downloaded once into a per-user folder. It is the fine-tuned
# DDR model saved without its optimizer state: identical predictions, half the
# size of best_model_ddr_finetuned_phase3.h5. No TensorFlow here — loading
# the file is cli.load_model's job.

import hashlib
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import certifi
from tqdm import tqdm

MODEL_FILENAME = "lungmap-model-v1.h5"
MODEL_URL = ("https://github.com/leojqian/LungSegmentation/releases/download/"
             f"model-v1/{MODEL_FILENAME}")
MODEL_SHA256 = "0e98c09089709c3a525e4407305edba8cba0d691c59cc1dc84885a78acdb88bf"
LEGACY_NAME = "best_model_ddr_finetuned_phase3.h5"   # the research checkout's own copy
MODEL_ENV = "LUNGMAP_MODEL"
URL_ENV = "LUNGMAP_MODEL_URL"                         # a mirror, or a local copy in tests


class ModelError(Exception):
    """The model can't be found or fetched; the message is meant for the user."""


def data_dir():
    """Per-user folder for the downloaded model, following each OS's convention."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "lungmap"


def find_model(flag=None):
    """-> the checkpoint to use, or None if it has to be downloaded first.

    --model, then $LUNGMAP_MODEL, then a research checkout's own copy in the
    working folder, then an earlier download. The first two are returned even
    if missing, so the caller can report the typo instead of silently
    downloading; None means fetch_model is needed.
    """
    if flag is not None:
        return Path(flag)
    if os.environ.get(MODEL_ENV):
        return Path(os.environ[MODEL_ENV])
    for candidate in (Path(LEGACY_NAME), data_dir() / MODEL_FILENAME):
        if candidate.is_file():
            return candidate
    return None


def fetch_model(progress=True):
    """Download the release checkpoint into data_dir(). -> its path. Raises ModelError."""
    print(f"First run: downloading the model (~126MB) to {data_dir()}")
    return download_model(data_dir() / MODEL_FILENAME, progress=progress)


def download_model(dest, url=None, sha256=None, progress=True):
    """Fetch the checkpoint to dest, verifying its SHA-256. -> dest.

    Written to dest + ".part" and renamed only once the checksum matches, so an
    interrupted or corrupted download never leaves a file that looks usable.
    """
    url = url or os.environ.get(URL_ENV) or MODEL_URL
    sha256 = sha256 or MODEL_SHA256
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    digest = hashlib.sha256()
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context, timeout=60) as response, \
                open(part, "wb") as f, \
                tqdm(total=int(response.headers.get("Content-Length") or 0) or None,
                     unit="B", unit_scale=True, desc="model", disable=not progress) as bar:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                digest.update(chunk)
                bar.update(len(chunk))
    except (urllib.error.URLError, OSError) as e:
        part.unlink(missing_ok=True)
        raise ModelError(f"could not download the model from {url} ({e}). "
                         "Check the internet connection and try again.") from e

    if digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise ModelError(f"downloaded model failed its checksum ({url}); try again")
    os.replace(part, dest)
    return dest
