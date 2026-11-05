#!/usr/bin/env bash
# Install project git hooks from .githooks/ into .git/hooks/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_SRC="$REPO_ROOT/.githooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

for hook in "$HOOKS_SRC"/*; do
    name="$(basename "$hook")"
    target="$HOOKS_DST/$name"

    chmod +x "$hook"

    if [[ -L "$target" ]]; then
        echo "Hook '$name' already installed (symlink) — skipping."
    elif [[ -f "$target" ]]; then
        echo "WARNING: '$target' already exists as a regular file — backing up to '$target.bak'."
        mv "$target" "$target.bak"
        ln -sf "$hook" "$target"
        echo "Hook '$name' installed (previous hook backed up)."
    else
        ln -sf "$hook" "$target"
        echo "Hook '$name' installed."
    fi
done

echo ""
echo "Git hooks installed from .githooks/"
