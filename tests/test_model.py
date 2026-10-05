# Tests for model.py — finding the U-Net checkpoint, and fetching it on first
# use. Downloads use file:// URLs, so no network is touched.

import hashlib
from pathlib import Path

import pytest

from lungmap import model as m


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No env var, an empty working folder, and a per-user cache inside tmp."""
    monkeypatch.delenv(m.MODEL_ENV, raising=False)
    monkeypatch.delenv(m.URL_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setattr(m, "data_dir", lambda: cache)
    return tmp_path


def published(tmp_path, payload=b"fake weights"):
    """A local file standing in for the release asset -> (file:// url, sha256)."""
    src = tmp_path / "release.h5"
    src.write_bytes(payload)
    return src.as_uri(), hashlib.sha256(payload).hexdigest()


class TestFindModel:
    def test_flag_wins(self, isolated):
        assert m.find_model(Path("x.h5")) == Path("x.h5")

    def test_env_var_next(self, isolated, monkeypatch):
        monkeypatch.setenv(m.MODEL_ENV, "env.h5")
        assert m.find_model(None) == Path("env.h5")

    def test_research_checkout_copy_in_the_working_folder(self, isolated):
        (isolated / m.LEGACY_NAME).write_bytes(b"x")
        assert m.find_model(None) == Path(m.LEGACY_NAME)

    def test_cached_download(self, isolated):
        cached = m.data_dir() / m.MODEL_FILENAME
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"x")
        assert m.find_model(None) == cached

    def test_none_when_it_must_be_downloaded(self, isolated):
        assert m.find_model(None) is None


class TestDownloadModel:
    def test_writes_the_file_when_the_hash_matches(self, isolated):
        url, sha = published(isolated)
        dest = isolated / "cache" / "model.h5"
        assert m.download_model(dest, url=url, sha256=sha, progress=False) == dest
        assert dest.read_bytes() == b"fake weights"
        assert not list(dest.parent.glob("*.part"))

    def test_hash_mismatch_raises_and_leaves_nothing_behind(self, isolated):
        url, _ = published(isolated)
        dest = isolated / "cache" / "model.h5"
        with pytest.raises(m.ModelError, match="checksum"):
            m.download_model(dest, url=url, sha256="0" * 64, progress=False)
        assert not dest.exists()
        assert not list(dest.parent.glob("*.part"))

    def test_unreachable_url_raises_a_readable_error(self, isolated):
        dest = isolated / "cache" / "model.h5"
        with pytest.raises(m.ModelError, match="could not download"):
            m.download_model(dest, url=(isolated / "missing.h5").as_uri(), progress=False)

    def test_url_env_var_overrides_the_release_url(self, isolated, monkeypatch):
        url, sha = published(isolated)
        monkeypatch.setenv(m.URL_ENV, url)
        dest = isolated / "cache" / "model.h5"
        m.download_model(dest, sha256=sha, progress=False)
        assert dest.exists()


class TestFetchModel:
    def test_downloads_into_the_per_user_cache(self, isolated, monkeypatch):
        url, sha = published(isolated)
        monkeypatch.setenv(m.URL_ENV, url)
        monkeypatch.setattr(m, "MODEL_SHA256", sha)
        path = m.fetch_model(progress=False)
        assert path == m.data_dir() / m.MODEL_FILENAME and path.exists()
        assert m.find_model(None) == path      # and the next run finds it
