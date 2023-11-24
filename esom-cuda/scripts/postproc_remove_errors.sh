#!/bin/bash

# remove error and blank lines from result CSV
# expects one argument -- a directory where all CSV files are fixed

cd `dirname "$0"` || exit 1

DIR=$1

if [ ! -d "$DIR" ]; then
	echo "Given path $DIR is not a directory."
	exit 1
fi

FILES=`echo $DIR/*.csv`
CHANGED=0
for FILE in $FILES; do
	echo -n "Removing errors from $FILE ... "

	ORIG_LINES=`cat "$FILE" | wc -l`
	TMP="${FILE}.tmp"
	mv "$FILE" "$TMP"
	grep -v -E -e 'Error|at [.]' "$TMP" | grep -v -E -e '^[[:space:]]*$' > "$FILE"
	rm "$TMP"
	NEW_LINES=`cat "$FILE" | wc -l`

	if [[ $ORIG_LINES != $NEW_LINES ]]; then
		DIFF=$(($ORIG_LINES - $NEW_LINES))
		echo "$DIFF LINES REMOVED !"
		CHANGED=$(($CHANGED+1))
	else
		echo "no errors"
	fi
done

echo
echo "Total $CHANGED files were modified."
