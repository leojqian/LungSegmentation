# MP4 output for the CLI's annotated frames — a quick-look companion to the
# overlay DICOM for anyone without a DICOM viewer.

import cv2


class Mp4Writer:
    """Frame-at-a-time MP4 writer; use as a context manager.

    MPEG-4 Part 2 ('mp4v') rather than H.264: pip's opencv-python can encode
    mp4v on both Windows and macOS out of the box, while H.264 needs an extra
    codec library on Windows. QuickTime, VLC and the Windows media apps all
    play it. Odd frame sizes are padded to even by edge replication, since the
    encoder's 4:2:0 chroma needs even dimensions.
    """

    def __init__(self, path, fps, size):
        width, height = size
        self.size = (width + width % 2, height + height % 2)
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                       float(fps), self.size)
        if not self._writer.isOpened():
            raise OSError(f"could not open {path} for video writing")

    def write(self, bgr):
        pad_x = self.size[0] - bgr.shape[1]
        pad_y = self.size[1] - bgr.shape[0]
        if pad_x or pad_y:
            bgr = cv2.copyMakeBorder(bgr, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)
        self._writer.write(bgr)

    def close(self):
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
