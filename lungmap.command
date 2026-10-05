#!/bin/bash
# lungmap for macOS.
#   Double-click this file (the very first time: right-click it > Open).
#   The first run downloads Python, the libraries and the model (about 1.5 GB,
#   several minutes). Later runs start in seconds.
#   Results go in a "lungmap_output" folder next to each scan, which opens when done.

cd "$(dirname "$0")" || exit 1

# Python environment and model live in ~/Library/Application Support/lungmap,
# not in this folder, so replacing it with a newer download keeps the install.
export UV_PROJECT_ENVIRONMENT="$HOME/Library/Application Support/lungmap/venv"
export PATH="$HOME/.local/bin:$PATH"

finish() {
    echo
    read -r -p "Press Enter to close this window." _
    exit "$1"
}

if ! command -v uv >/dev/null 2>&1; then
    echo "First run: installing uv, which sets up Python for lungmap ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Could not install uv."; finish 1; }
fi

if [ $# -eq 0 ]; then
    echo "Drag a DICOM file or a folder into this window, then press Enter:"
    # No -r on purpose: Terminal backslash-escapes the spaces in dropped paths,
    # and plain read removes those escapes while splitting one path per word.
    read -a inputs
    set -- "${inputs[@]}"
fi
[ $# -gt 0 ] || { echo "Nothing to do."; finish 0; }

echo
uv run --locked lungmap --open "$@"
status=$?
[ $status -eq 0 ] && echo "Done." || echo "Something went wrong - see the messages above."
finish $status
