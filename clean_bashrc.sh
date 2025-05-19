#!/bin/bash

BASHRC="$HOME/.bashrc"
TEMP_FILE="$(mktemp)"

awk '
/^# >>> conda initialize >>>/ { skip=1 }
/^# <<< conda initialize <<</ { skip=0; next }
!skip
' "$BASHRC" > "$TEMP_FILE"

mv "$TEMP_FILE" "$BASHRC"
echo "conda initialization block removed from $BASHRC."
echo "Current $BASHRC:"
cat $BASHRC
