# Tests for helpers in the research scripts (research/, not part of the lungmap
# package). The script folder is put on sys.path the same way Python does when
# the script itself is run.

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "montgomery"))
from step05_map_diaphragm import select_sample, stem  # noqa: E402


class TestStem:
    def test_strips_directory_and_extension(self):
        assert stem("MontgomerySet/CXR_png/MCUCXR_0001_0.png") == "MCUCXR_0001_0"


class TestSelectSample:
    PAIRS = [
        ("cxr/MCUCXR_0001_0.png", "l/MCUCXR_0001_0.png", "r/MCUCXR_0001_0.png"),
        ("cxr/MCUCXR_0035_0.png", "l/MCUCXR_0035_0.png", "r/MCUCXR_0035_0.png"),
    ]

    def test_finds_by_bare_name(self):
        assert select_sample(self.PAIRS, "MCUCXR_0035_0") == [self.PAIRS[1]]

    def test_finds_by_filename_with_extension(self):
        assert select_sample(self.PAIRS, "MCUCXR_0035_0.png") == [self.PAIRS[1]]

    def test_unknown_name_raises_with_the_name_in_the_message(self):
        with pytest.raises(SystemExit, match="MCUCXR_9999_9"):
            select_sample(self.PAIRS, "MCUCXR_9999_9")
