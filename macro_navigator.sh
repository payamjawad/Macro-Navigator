#!/bin/bash
# Macro Navigator launcher logic. "Run Macro Navigator.command" calls this file, so updates to
# this file never need macOS approval again.
#
# It finds a Python environment that can run the data libraries without crashing, trying in order:
#   1. the environment that worked last time
#   2. your normal python3
#   3. a fresh private environment built from your python3
#   4. a private Python 3.12 downloaded with uv (stable, fully separate from your system)
#   5. the same without pyarrow (files saved as compressed CSV instead of parquet)
# Each candidate must pass a quick test in a separate process before it is used.

set -u
set -o pipefail
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT" || exit 1
STORE="$ROOT/data/store"
LOG="$ROOT/data/last_run.log"
BASE="$HOME/.macro_navigator"
KC_SERVICE="macro-navigator-fred"
PKGS="pandas numpy requests yfinance pyarrow xlrd openpyxl lxml"
PKGS_LITE="pandas numpy requests yfinance xlrd openpyxl lxml"
mkdir -p "$STORE" "$BASE"
[ -z "${MN_RETRIED:-}" ] && : > "$LOG"   # keep the first attempt's log on an automatic retry

say_done() { echo; read -n 1 -s -r -p "Press any key to close this window."; echo; }
dialog() { osascript -e "display dialog \"$1\" with title \"Macro Navigator\" buttons {\"OK\"} default button \"OK\"" >/dev/null 2>&1; }
log() { echo "$*" | tee -a "$LOG"; }

# --- install an update dropped into Downloads (unzipped MacroNavigator-update folder), then restart
if [ -z "${MN_UPDATED:-}" ]; then
  for UPD in "$HOME/Downloads"/MacroNavigator-update*/; do
    [ -d "$UPD" ] || continue
    echo "Installing update from ${UPD%/}..."
    DEST_PY="$ROOT"; [ -d "$ROOT/scripts" ] && DEST_PY="$ROOT/scripts"
    cp -f "$UPD"*.py "$DEST_PY/" 2>/dev/null
    cp -f "$UPD"macro_navigator.sh "$ROOT/" 2>/dev/null
    rm -rf "${UPD%/}" "${UPD%/}.zip"
    MN_UPDATED=1 exec bash "$ROOT/macro_navigator.sh" "$ROOT"
  done
fi

log "Macro Navigator · $(date '+%a %d %b %Y %H:%M')"
echo

# --- scripts live next to this file or in a scripts folder
if [ -f "$ROOT/scripts/fetch_all.py" ]; then S="$ROOT/scripts"; else S="$ROOT"; fi
if [ ! -f "$S/fetch_all.py" ] || [ ! -f "$S/evidence_check.py" ]; then
  log "fetch_all.py and evidence_check.py must sit next to this file."
  dialog "fetch_all.py and evidence_check.py were not found next to the launcher."
  say_done; exit 1
fi

SYS_PY="$(command -v python3 || true)"
if [ -z "$SYS_PY" ]; then
  dialog "Python 3 is not installed. Install it from python.org, then double-click again."
  open "https://www.python.org/downloads/macos/"; exit 1
fi

# --- the test every environment must pass (exercises the code paths that crashed before)
SMOKE='import io, os, platform, tempfile
import numpy as np, pandas as pd
df = pd.read_csv(io.StringIO("name,value\nalpha,1\nbeta,2\n"))
df["upper"] = df["name"].str.upper()
pd.to_datetime(pd.Series(["2020-01-31", "2021-06-30"]))
df.groupby("name")["value"].sum()
try:
    import pyarrow
except ImportError:
    pyarrow = None
if pyarrow is not None:
    p = os.path.join(tempfile.gettempdir(), "mn_smoke_%d.parquet" % os.getpid())
    df.to_parquet(p); pd.read_parquet(p); os.remove(p)
import requests, yfinance, openpyxl, xlrd
print("ok Python", platform.python_version(), "· pandas", pd.__version__, "· numpy", np.__version__,
      "· pyarrow", getattr(pyarrow, "__version__", "not used"))'

PY=""; INFO=""
BLOCKED="$BASE/blocked_v2"
touch "$BLOCKED"
is_blocked() { grep -qxF "$1" "$BLOCKED" 2>/dev/null; }
try_env() {
  local py="$1" out rc
  [ -n "$py" ] && [ -x "$py" ] || return 1
  if is_blocked "$py"; then echo "  skipping $py (crashed during an earlier run)" >> "$LOG"; return 1; fi
  out="$("$py" -X faulthandler -c "$SMOKE" 2>>"$LOG")"; rc=$?
  if [ $rc -eq 0 ] && [[ "$out" == ok* ]]; then PY="$py"; INFO="${out#ok }"; echo "$py" > "$BASE/chosen"; return 0; fi
  if [ $rc -ge 128 ]; then echo "  environment test crashed ($py, signal $((rc - 128)))" >> "$LOG"
  else echo "  environment test failed ($py, code $rc)" >> "$LOG"; fi
  return 1
}

build_venv() {  # $1 python, $2 target, $3 packages
  is_blocked "$2/bin/python" && return 1
  echo "  Setting up a clean private environment (one time, about a minute)..."
  rm -rf "$2"
  "$1" -m venv "$2" >>"$LOG" 2>&1 || return 1
  "$2/bin/python" -m pip install -q --upgrade pip >>"$LOG" 2>&1
  "$2/bin/python" -m pip install -q --no-cache-dir $3 >>"$LOG" 2>&1
}

build_uv() {  # $1 target, $2 packages: private Python 3.12 managed by uv
  is_blocked "$1/bin/python" && return 1
  if [ -x "$1/bin/python" ] && "$1/bin/python" -c "import pandas" 2>/dev/null; then return 0; fi   # already built
  echo "  Downloading a separate, stable Python 3.12 (one time, 1-2 minutes)..."
  local tools="$BASE/tools"
  if [ ! -x "$tools/bin/python" ]; then "$SYS_PY" -m venv "$tools" >>"$LOG" 2>&1 || return 1; fi
  "$tools/bin/python" -m pip install -q uv >>"$LOG" 2>&1 || return 1
  rm -rf "$1"
  "$tools/bin/python" -m uv venv -q --python 3.12 --python-preference only-managed "$1" >>"$LOG" 2>&1 || return 1
  "$tools/bin/python" -m uv pip install -q --python "$1/bin/python" $2 >>"$LOG" 2>&1
}

echo "Checking the Python environment..."
LAST="$(cat "$BASE/chosen" 2>/dev/null || true)"
if try_env "$LAST"; then :
elif try_env "$BASE/py312/bin/python"; then :
elif try_env "$SYS_PY"; then :
elif build_venv "$SYS_PY" "$BASE/py_fresh" "$PKGS" && try_env "$BASE/py_fresh/bin/python"; then :
elif build_uv "$BASE/py312" "$PKGS" && try_env "$BASE/py312/bin/python"; then :
elif build_uv "$BASE/py312_lite" "$PKGS_LITE" && try_env "$BASE/py312_lite/bin/python"; then :
else
  log "No working Python environment could be set up. Details are in $LOG"
  dialog "Could not set up a working Python environment. Please send the file data/last_run.log to your assistant."
  open -R "$LOG" 2>/dev/null
  say_done; exit 1
fi
log "  Using $INFO"
echo

# --- FRED key: environment, then Keychain, then ask once
KEY="${FRED_API_KEY:-}"
[ -n "$KEY" ] || KEY="$(security find-generic-password -a "$USER" -s "$KC_SERVICE" -w 2>/dev/null || true)"
if [ -z "$KEY" ]; then
  KEY="$(osascript \
    -e 'set r to display dialog "Paste your free FRED API key (fred.stlouisfed.org → My Account → API Keys). It is stored in your Mac Keychain and asked for only once. Choose Skip to use FRED'\''s slower public download instead." default answer "" with title "Macro Navigator" buttons {"Skip", "Save"} default button "Save"' \
    -e 'if button returned of r is "Save" then return text returned of r' 2>/dev/null || true)"
  KEY="$(printf '%s' "$KEY" | tr -d '[:space:]')"
  if [ -n "$KEY" ]; then
    if [[ "$KEY" =~ ^[a-z0-9]{32}$ ]]; then
      security add-generic-password -a "$USER" -s "$KC_SERVICE" -w "$KEY" -U >/dev/null 2>&1 && echo "FRED key saved to your Keychain."
    else
      dialog "That does not look like a FRED key (32 lowercase letters and digits). Running without it this time."
      KEY=""
    fi
  fi
fi
export FRED_API_KEY="$KEY"

# --- 1. fetch
MARK="$BASE/.run_marker"; touch "$MARK"
echo "Step 1 of 3 · fetching data"
"$PY" -X faulthandler -u "$S/fetch_all.py" --out "$STORE" 2>&1 | tee -a "$LOG"
FETCH_STATUS=$?
if [ -n "$KEY" ] && grep -q "api_key" "$STORE/catalog.csv" 2>/dev/null; then
  security delete-generic-password -a "$USER" -s "$KC_SERVICE" >/dev/null 2>&1
  dialog "FRED rejected the saved key, so it was removed. Double-click again and paste a valid key."
fi
FETCH_OK=0; [ -f "$STORE/catalog.csv" ] && [ "$STORE/catalog.csv" -nt "$MARK" ] && FETCH_OK=1

# --- 2. evidence and report
echo
echo "Step 2 of 3 · checking the evidence"
rm -f "$STORE/evidence_report.html"        # only ever open a report made in this run
"$PY" -X faulthandler -u "$S/evidence_check.py" --store "$STORE" 2>&1 | tee -a "$LOG"
EVID_STATUS=$?
EVID_OK=0; [ -f "$STORE/evidence_report.html" ] && EVID_OK=1
CRASH_SIGNS=0
{ [ "$FETCH_STATUS" -ge 128 ] || [ "$EVID_STATUS" -ge 128 ] || grep -qiE "segmentation fault|Fatal Python error" "$LOG"; } && CRASH_SIGNS=1
CRASHED=0
if [ "$CRASH_SIGNS" -eq 1 ]; then
  if [ "$FETCH_OK" -eq 1 ] && [ "$EVID_OK" -eq 1 ]; then
    echo; echo "Note: Python crashed while closing, after every result was saved. Your results are complete."
  else
    CRASHED=1
  fi
fi
if [ "$CRASHED" -eq 1 ]; then
  echo "$PY" >> "$BLOCKED"; rm -f "$BASE/chosen"
  if [ -z "${MN_RETRIED:-}" ]; then
    echo
    echo "This Python setup crashed, so it is now blocked. Retrying automatically with a more isolated one..."
    echo
    MN_RETRIED=1 exec bash "$0" "$ROOT"
  fi
  echo "The retry crashed as well. Please send data/last_run.log to your assistant."
fi

# --- 3. market engine (only when the data and evidence steps worked)
ENGINE="$STORE/engine_report.html"
rm -f "$ENGINE"
if [ "$EVID_OK" -eq 1 ] && [ -f "$S/engine.py" ]; then
  echo
  echo "Step 3 of 3 · running the market engine"
  "$PY" -X faulthandler -u "$S/engine.py" --store "$STORE" 2>&1 | tee -a "$LOG"
fi

REPORT="$STORE/evidence_report.html"
MAIN="$ENGINE"; [ -f "$MAIN" ] || MAIN="$REPORT"
if [ -f "$MAIN" ]; then
  open "$MAIN"
  # phone access: a copy in iCloud Drive opens in the Files app on iPhone (only if iCloud Drive is set up)
  ICLOUD="$HOME/Library/Mobile Documents/com~apple~CloudDocs"
  if [ -d "$ICLOUD" ]; then
    mkdir -p "$ICLOUD/MacroNavigator" 2>/dev/null
    cp -f "$MAIN" "$ICLOUD/MacroNavigator/Macro Navigator.html" 2>/dev/null
    [ -f "$REPORT" ] && cp -f "$REPORT" "$ICLOUD/MacroNavigator/evidence_report.html" 2>/dev/null
    echo "Copied to iCloud Drive › MacroNavigator (open it on iPhone in the Files app)."
  fi
  osascript -e 'display notification "Macro Navigator is open in your browser." with title "Macro Navigator"' >/dev/null 2>&1
  echo; echo "Opened: $MAIN"
else
  echo "No report was produced. The full log is in $LOG"
fi
[ "$FETCH_STATUS" -eq 0 ] || echo "The fetch step reported a problem; details are in $LOG"
say_done
