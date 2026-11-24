#!/bin/bash
# Double-click to run Macro Navigator. All the logic lives in macro_navigator.sh beside this file,
# so updates never require approving this launcher again. Drag this file to the Dock for one click.
DIR="$(cd "$(dirname "$0")" && pwd)"
for f in "$DIR/macro_navigator.sh" "$DIR/scripts/macro_navigator.sh"; do
  if [ -f "$f" ]; then exec bash "$f" "$DIR"; fi
done
echo "macro_navigator.sh was not found next to this file."
read -n 1 -s -r -p "Press any key to close this window."
