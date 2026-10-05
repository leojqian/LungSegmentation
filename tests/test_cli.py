# End-to-end tests for the `lungmap` command: a synthetic multi-frame DICOM in,
# the points CSV + overlay DICOM + MP4 out. The U-Net is replaced by FakeModel
# (monkeypatched over model_file.load_model) so no TensorFlow or weights are needed.

import csv
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
import pytest

from fakes import FakeModel, write_dicom
from lungmap import cli, model_file
from lungmap.points import CSV_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent
N_FRAMES = 4


@pytest.fixture
def dicom(tmp_path):
    path = tmp_path / "case one.dcm"     # a space, as Windows/macOS paths often have
    frames = np.random.default_rng(0).integers(0, 4000, (N_FRAMES, 400, 400)).astype(np.uint16)
    write_dicom(str(path), frames, FrameTime=66, PatientID="P001")
    return path


@pytest.fixture
def fake_weights(tmp_path, monkeypatch):
    path = tmp_path / "model.h5"
    path.write_bytes(b"not really a model")
    monkeypatch.setattr(model_file, "load_model", lambda p: FakeModel())
    return path


@pytest.fixture
def run(dicom, fake_weights, tmp_path):
    out = tmp_path / "out"
    code = cli.main([str(dicom), "-o", str(out), "--model", str(fake_weights)])
    return code, out


class TestOutputs:
    def test_exits_zero_and_writes_the_three_files(self, run):
        code, out = run
        assert code == 0
        assert sorted(p.name for p in out.iterdir()) == [
            "case one_overlay.dcm", "case one_overlay.mp4", "case one_points.csv"]

    def test_csv_has_a_header_and_one_row_per_frame(self, run):
        _, out = run
        with open(out / "case one_points.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == CSV_COLUMNS
        assert [r[0] for r in rows[1:]] == [str(i) for i in range(N_FRAMES)]
        assert all(len(r) == 17 for r in rows)

    def test_both_lungs_are_measured_on_every_frame(self, run):
        _, out = run
        with open(out / "case one_points.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row["R_apex_x"] and row["L_apex_x"]
            assert float(row["R_apex_x"]) < float(row["L_apex_x"])   # R = image-left

    def test_overlay_dicom_has_every_frame_at_full_resolution(self, run):
        _, out = run
        ds = pydicom.dcmread(out / "case one_overlay.dcm")
        assert ds.NumberOfFrames == N_FRAMES
        assert (ds.Rows, ds.Columns) == (400, 400)
        assert ds.PatientID == "P001"

    def test_mp4_has_every_frame(self, run):
        _, out = run
        cap = cv2.VideoCapture(str(out / "case one_overlay.mp4"))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N_FRAMES
        cap.release()

    def test_prints_the_naive_vs_algorithm_comparison(self, dicom, fake_weights, tmp_path, capsys):
        cli.main([str(dicom), "-o", str(tmp_path / "o"), "--model", str(fake_weights)])
        printed = capsys.readouterr().out
        assert "naive" in printed and "R:" in printed and "L:" in printed
        printed.encode("ascii")   # legacy Windows consoles can't print anything else


class TestErrors:
    def test_missing_input_exits_2(self, fake_weights, tmp_path):
        with pytest.raises(SystemExit) as e:
            cli.main([str(tmp_path / "nope.dcm"), "--model", str(fake_weights)])
        assert e.value.code == 2

    def test_non_dicom_input_exits_2(self, fake_weights, tmp_path):
        bogus = tmp_path / "notes.dcm"
        bogus.write_text("hello")
        with pytest.raises(SystemExit) as e:
            cli.main([str(bogus), "--model", str(fake_weights)])
        assert e.value.code == 2

    def test_explicit_model_path_that_does_not_exist_exits_2(self, dicom, tmp_path):
        with pytest.raises(SystemExit) as e:
            cli.main([str(dicom), "--model", str(tmp_path / "typo.h5")])
        assert e.value.code == 2

    def test_failed_model_download_exits_1_with_a_message(self, dicom, tmp_path, monkeypatch,
                                                          capsys):
        monkeypatch.delenv(model_file.MODEL_ENV, raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(model_file, "data_dir", lambda: tmp_path / "cache")
        monkeypatch.setenv(model_file.URL_ENV, (tmp_path / "no_such_release.h5").as_uri())
        assert cli.main([str(dicom)]) == 1
        assert "could not download" in capsys.readouterr().err

    def test_one_bad_file_does_not_stop_the_others(self, dicom, fake_weights, tmp_path,
                                                   monkeypatch):
        second = tmp_path / "second.dcm"
        second.write_bytes(dicom.read_bytes())
        real = cli.process_dicom

        def flaky(path, *a, **kw):
            if Path(path).name == "case one.dcm":
                raise ValueError("boom")
            return real(path, *a, **kw)

        monkeypatch.setattr(cli, "process_dicom", flaky)
        out = tmp_path / "out"
        assert cli.main([str(dicom), str(second), "-o", str(out), "--model", str(fake_weights)]) == 1
        assert (out / "second_points.csv").exists()


class TestInputs:
    """What your dad will actually do: drop a folder, or use a wildcard on Windows."""

    def test_default_output_goes_next_to_the_input(self, dicom, fake_weights):
        assert cli.main([str(dicom), "--model", str(fake_weights)]) == 0
        assert (dicom.parent / cli.OUT_DIR_NAME / "case one_points.csv").exists()

    def test_a_folder_processes_every_dicom_inside_it(self, dicom, fake_weights, tmp_path):
        folder = tmp_path / "scans"
        folder.mkdir()
        for name in ("a.dcm", "b"):                          # DICOMs need no extension
            (folder / name).write_bytes(dicom.read_bytes())
        (folder / "notes.txt").write_text("not a scan")
        (folder / "old_overlay.dcm").write_bytes(dicom.read_bytes())   # our own output
        out = tmp_path / "out"
        assert cli.main([str(folder), "-o", str(out), "--model", str(fake_weights)]) == 0
        assert sorted(p.name for p in out.glob("*_points.csv")) == ["a_points.csv",
                                                                     "b_points.csv"]

    def test_wildcards_are_expanded_even_when_the_shell_does_not(self, dicom, fake_weights,
                                                                 tmp_path):
        out = tmp_path / "out"
        pattern = str(dicom.parent / "*.dcm")                # passed literally, as on Windows
        assert cli.main([pattern, "-o", str(out), "--model", str(fake_weights)]) == 0
        assert (out / "case one_points.csv").exists()

    def test_folder_without_dicoms_exits_2(self, fake_weights, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SystemExit) as e:
            cli.main([str(empty), "--model", str(fake_weights)])
        assert e.value.code == 2

    def test_open_shows_each_output_folder_once(self, dicom, fake_weights, tmp_path,
                                                monkeypatch):
        opened = []
        monkeypatch.setattr(cli, "open_folder", opened.append)
        second = dicom.parent / "second.dcm"
        second.write_bytes(dicom.read_bytes())
        cli.main([str(dicom), str(second), "--open", "--model", str(fake_weights)])
        assert opened == [dicom.parent / cli.OUT_DIR_NAME]


class TestEntryPoint:
    def test_python_dash_m_help(self):
        result = subprocess.run([sys.executable, "-m", "lungmap", "--help"], cwd=REPO_ROOT,
                                capture_output=True, text=True)
        assert result.returncode == 0
        assert "DICOM" in result.stdout

    def test_importing_the_cli_does_not_import_tensorflow(self):
        # TF takes seconds to import; --help and argument errors must not wait for it.
        code = "import sys, lungmap.cli; print('tensorflow' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                                capture_output=True, text=True, env={**os.environ})
        assert result.stdout.strip() == "False"
