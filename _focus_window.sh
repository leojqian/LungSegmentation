# Bring a background-launched GUI process to the front on macOS. Sourced by
# compare_predictions.sh / slideshow_predictions.sh, not run directly.
#
# This venv's Python (pyenv, non-framework build) creates real matplotlib
# windows, but macOS doesn't register the process as a normal foreground app
# -- no bundle/Dock identity -- so `System Events` (UI scripting) can't find
# it to activate it, and the window can open behind other windows with
# keyboard focus going nowhere. NSRunningApplication (used here via JXA)
# finds it anyway: any process connected to the WindowServer gets an entry
# there regardless of bundle status.
activate_pid() {
    local pid="$1"
    command -v osascript >/dev/null 2>&1 || return 0
    for _ in 1 2 3 4 5 6 7 8; do
        sleep 1
        result=$(osascript -l JavaScript -e "
            ObjC.import('AppKit');
            var app = \$.NSRunningApplication.runningApplicationWithProcessIdentifier($pid);
            if (!app.isNil()) {
                app.activateWithOptions(\$.NSApplicationActivateIgnoringOtherApps);
                'ok';
            }
        " 2>/dev/null)
        if [ "$result" = "ok" ]; then
            return 0
        fi
    done
}
