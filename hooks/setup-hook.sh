#!/usr/bin/env bash
set -e

ROOT=$(git rev-parse --show-toplevel)

if [ -f $ROOT/.git/hooks/commit-msg ];
then
    echo "File .git/hooks/commit-msg already exists! Will not attempt to overwrite."
    exit 1
fi

cp $ROOT/hooks/commit-msg $ROOT/.git/hooks/commit-msg
chmod +x $ROOT/.git/hooks/commit-msg

