#!/usr/bin/env bash
TOTAL_BYTES_1=$(wc -c $1 | cut -d " " -f1)
TOTAL_BYTES_2=$(wc -c $2 | cut -d " " -f1)

if [ $TOTAL_BYTES_1 != $TOTAL_BYTES_2 ]; then
    echo "Rulesets have different lengths."
    exit 1
fi

DIFF_BYTES=$(cmp -l $1 $2 | wc -l)
DIFF_PERCENTAGE=$(bc -l <<< "scale=2; 100 * $DIFF_BYTES / $TOTAL_BYTES_1")

echo "Rules total:     $TOTAL_BYTES_1"
echo "Rules different: $DIFF_BYTES"
echo "Difference:      $DIFF_PERCENTAGE%"
