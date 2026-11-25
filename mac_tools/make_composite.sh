#!/bin/bash
# One-off: writes config/composite.json for the forward test, from the idea factory's discovery results on this Mac.
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$(cat "$HOME/.macro_navigator/chosen" 2>/dev/null)"
[ -x "$PY" ] || PY="$(command -v python3)"
export SEC_CONTACT="$(cat "$HOME/.macro_navigator/sec_contact" 2>/dev/null)"
caffeinate -dims "$PY" "$DIR/export_composite.py"
