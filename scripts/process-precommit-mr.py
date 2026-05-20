#!/usr/bin/env python3
"""Post a GitLab MR note after a pre-commit failure.

Required environment variables:
- CI_PROJECT_ID: Numeric GitLab project identifier.
- CI_MERGE_REQUEST_IID: Merge request IID in the current project.
- CI_API_V4_URL: Base URL for GitLab API v4 (for example, https://gitlab.example.com/api/v4).
- CI_JOB_TOKEN: Token with permission to create MR notes.

Diagnostics are logged to stderr. Exit code 0 means success or graceful skip; non-zero indicates an error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence
import urllib.error
import urllib.parse
import urllib.request


def log_info(msg: str) -> None:
    """Log an informational message to stderr."""
    print(f"[precommit-mr] {msg}", file=sys.stderr)


def log_error(msg: str) -> None:
    """Log an error message to stderr."""
    print(f"[precommit-mr] ERROR: {msg}", file=sys.stderr)


# Environment variable names
PROJECT_ID = "CI_PROJECT_ID"
MR_IID = "CI_MERGE_REQUEST_IID"
API_URL = "CI_API_V4_URL"
PROJECT_URL = "CI_PROJECT_URL"
JOB_ID = "CI_JOB_ID"
TOKEN = "VECGEOM_GITLAB_TOKEN"  # see https://gitlab.cern.ch/VecGeom/VecGeom/-/settings/ci_cd#js-cicd-variables-settings
REQUIRED_ENV_VARS = {PROJECT_ID, MR_IID, API_URL, PROJECT_URL, JOB_ID, TOKEN}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post a merge request note when pre-commit fails in CI.",
        epilog=("Required env vars: " + ", ".join(REQUIRED_ENV_VARS)),
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="List of changed files.",
    )
    parser.add_argument(
        "--fixes-patch",
        type=Path,
        default=None,
        help="Path to the generated patch file artifact. Omit when no autofix is available.",
    )
    parser.add_argument(
        "--log-tail-lines",
        type=int,
        default=120,
        help="Number of pre-commit log lines to include in the MR note.",
    )
    return parser.parse_args(argv)


def build_comment_body(
    files: list[str], env: dict[str, str], fixes_patch: Optional[Path]
) -> str:
    changed_count = len(files)
    file_list = "\n".join(f"- {path}" for path in files)

    header = (
        "**Pre-commit failed** on files in this MR. See CI output for more details."
    )
    if fixes_patch is None:
        return f"{header}\n\n**No autofix available.** Please fix the issue manually and push again."

    artifact_path = urllib.parse.quote(str(fixes_patch), safe="/")
    artifact_url = f"{env[PROJECT_URL]}/-/jobs/{env[JOB_ID]}/artifacts/raw/{artifact_path}?inline=false"
    return f"""\
{header}

Fixed files ({changed_count}):
{file_list}

**Download and apply** [this patch]({artifact_url}) to fix:
```console
$ git am {artifact_path}
$ git push
```
"""


def post_merge_request_note(env, body: str) -> int:
    """Post a note to a GitLab merge request via the API."""
    payload = json.dumps({"body": body}).encode("utf-8")
    url = (
        f"{env[API_URL]}/projects/{env[PROJECT_ID]}/merge_requests/{env[MR_IID]}/notes"
    )
    log_info(f"Posting MR note to {url}")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"PRIVATE-TOKEN": str(env[TOKEN]), "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            status = response.status
            log_info(f"MR note posted successfully (HTTP {status})")
            return 0
    except urllib.error.HTTPError as e:
        log_error(f"HTTP error posting MR note: {e.code} {e.reason}")
        return 1
    except urllib.error.URLError as e:
        log_error(f"URL error posting MR note: {e.reason}")
        return 1
    except Exception as e:
        log_error(f"Unexpected error posting MR note: {e}")
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Check for missing env vars using set operations.
    missing = {var for var in REQUIRED_ENV_VARS if var not in os.environ}

    if missing:
        log_error(
            f"Missing required env vars: {', '.join(sorted(missing))}. Skipping MR note."
        )
        return 1

    # Load all required environment variables into a dict.
    env = {var: os.environ[var] for var in REQUIRED_ENV_VARS}

    try:
        body = build_comment_body(args.files, env, args.fixes_patch)
        return post_merge_request_note(env, body)
    except Exception as e:
        log_error(f"Unexpected error in main: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
