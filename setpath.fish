#
# This script adds Mitsuba to the current path.
# It works for the Fish shell.
#
# NOTE: this script must be sourced and not run, i.e.
#    . setpath.fish
#

set -x MITSUBA_DIR (dirname (realpath (status -f)))

if [ -n "$MITSUBA_PYVER" ]
	set pyver $MITSUBA_PYVER
else
	set pyver (python --version 2>&1 | grep -oE '([[:digit:]].[[:digit:]])')
end

if [ (uname) = 'Darwin' ]
	set -x PYTHONPATH "$MITSUBA_DIR/Mitsuba.app/python/$pyver" $PYTHONPATH
else
	set -x PYTHONPATH "$MITSUBA_DIR/dist/python" "$MITSUBA_DIR/dist/python/$pyver" $PYTHONPATH
end

if [ (uname) = 'Darwin' ]
	set -x PATH "$MITSUBA_DIR/Mitsuba.app/Contents/MacOS" $PATH
else
	set -x LD_LIBRARY_PATH "$MITSUBA_DIR/dist" $LD_LIBRARY_PATH
	set -x PATH "$MITSUBA_DIR/dist" $PATH
end

