# Quick viewer for DICOM frames. Usage:
#   python view_dicom.py path/to/file.dcm [start_frame_index]
#
# Drag the slider, or click the image and use Left/Right arrows or the
# scroll wheel, to step through frames.

import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from formats.dicom_io import load_frames, to_model_input


def main():
    path = sys.argv[1]
    frame_idx = [int(sys.argv[2]) if len(sys.argv) > 2 else 0]

    frames, spacing_mm, photometric = load_frames(path)
    n = len(frames)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)
    im = ax.imshow(to_model_input(frames[frame_idx[0]], photometric), cmap="gray")
    ax.axis("off")

    slider_ax = fig.add_axes([0.2, 0.03, 0.6, 0.04])
    slider = Slider(slider_ax, "Frame", 0, n - 1, valinit=frame_idx[0], valstep=1)

    def render():
        im.set_data(to_model_input(frames[frame_idx[0]], photometric))
        ax.set_title(f"{path}\nframe {frame_idx[0]}/{n - 1} ({spacing_mm} mm/px)")
        fig.canvas.draw_idle()

    def step(delta):
        frame_idx[0] = (frame_idx[0] + delta) % n
        slider.eventson = False
        slider.set_val(frame_idx[0])
        slider.eventson = True
        render()

    def on_slider(val):
        frame_idx[0] = int(val)
        render()

    def on_key(event):
        if event.key == "right":
            step(1)
        elif event.key == "left":
            step(-1)

    def on_scroll(event):
        step(1 if event.button == "up" else -1)

    slider.on_changed(on_slider)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    render()
    plt.show()


if __name__ == "__main__":
    main()
