#!/bin/sh

if ! git_hook_dir="$(git rev-parse --git-path hooks 2>/dev/null)"; then
  printf "Not in a git repository: cannot set up git hooks" >&2
  exit 1
fi

if ! command -v pre-commit >/dev/null 2>&1; then
  printf "\e[31mpre-commit is not installed.\e[0m
Install pre-commit and update your PATH.
" >&2
  for pgm in brew pip yum dnf; do
    if command -v "$pgm" >/dev/null 2>&1 ; then
      printf "\e[33;40mmaybe:\e[0m %s install pre-commit\n" "$pgm">&2
    fi
  done
  printf "See \e[34;40mhttps://pre-commit.com\e[0m\n" >&2
  exit 1
fi

for script in pre-commit post-commit commit-msg ; do
  FILENAME="${git_hook_dir}/${script}"
  if grep -l "scripts/dev" "$FILENAME" >/dev/null 2>&1; then
    printf "\e[31mDisabling obsolete %s hook at %s.\e[0m\n" "$script" "$FILENAME" >&2
    mv "$FILENAME" "$FILENAME.disabled"
  fi
done

pre-commit install --install-hooks
