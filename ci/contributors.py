#!/usr/bin/env python3
"""Check commit authors against LitKit's contributor list.

    python3 ci/contributors.py check --pr N
    python3 ci/contributors.py check --all

GitHub supplies the account associated with each commit. For unlinked commits,
use the recorded name or a known historical identity. The API lists up to 250
commits per PR. Both modes check primary authors and skip bot accounts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRIBUTORS = ROOT / "CONTRIBUTORS.md"
DEFAULT_REPO = "lanl/litkit"

# GitHub did not link these older commits. Both addresses appear in LitKit's
# history under the same author's name. Use the public handle for credit.
HISTORICAL_AUTHORS = {("bill hlavacek", "wish@lanl.gov"): "wshlavacek"}

#: One entry of CONTRIBUTORS.md: ``- Ada Lovelace (@alovelace)``, or ``- @alovelace``
#: for handle only. GitHub handles are letters, digits and single hyphens.
ENTRY = re.compile(
    r"^- (?:(?P<name>\S.*?) \(@(?P<handle>[A-Za-z0-9-]+)\)|@(?P<bare>[A-Za-z0-9-]+))\s*$"
)

#: GitHub returns at most this many commits of a pull request.
PR_COMMITS_LIMIT = 250


@dataclass(frozen=True)
class Commit:
    sha: str
    login: str  # "" where GitHub matched the email to no account
    account_type: str  # "User", "Bot", ... ; "" with no account
    name: str
    email: str


def listed(text: str) -> tuple[set[str], set[str]]:
    """The handles and the names in CONTRIBUTORS.md, both casefolded.

    GitHub handles are case-insensitive, and a name is compared only where GitHub
    knows no handle, so case is not what tells two people apart.
    """
    handles: set[str] = set()
    names: set[str] = set()
    for line in text.splitlines():
        match = ENTRY.match(line)
        if match:
            handles.add((match["handle"] or match["bare"]).casefold())
            if match["name"]:
                names.add(match["name"].strip().casefold())
    return handles, names


def unlisted(commits: list[Commit], handles: set[str], names: set[str]) -> list[Commit]:
    """The commits whose author CONTRIBUTORS.md does not list, one per author."""
    missing: dict[str, Commit] = {}
    for commit in commits:
        if commit.account_type == "Bot" or commit.login.endswith("[bot]"):
            continue
        if commit.login:
            if commit.login.casefold() in handles:
                continue
            key = "@" + commit.login.casefold()
        else:
            # A git config user.name is often the person's handle.
            name = commit.name.strip().casefold()
            historical_login = HISTORICAL_AUTHORS.get((name, commit.email.casefold()))
            if name in names or name in handles or historical_login in handles:
                continue
            key = commit.email.casefold()
        missing.setdefault(key, commit)
    return list(missing.values())


def _gh_commits(endpoint: str) -> list[Commit]:
    """Every commit an API list endpoint returns, across its pages."""
    jq = (
        '.[] | [.sha, (.author.login // ""), (.author.type // ""),'
        ' (.commit.author.name // ""), (.commit.author.email // "")]'
    )
    try:
        done = subprocess.run(
            ["gh", "api", "--paginate", endpoint, "--jq", jq],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("contributors: needs the GitHub CLI (gh) to read commits", file=sys.stderr)
        sys.exit(2)
    if done.returncode != 0:
        print(f"contributors: gh api {endpoint} failed:\n{done.stderr.strip()}", file=sys.stderr)
        sys.exit(2)
    return [Commit(*json.loads(line)) for line in done.stdout.splitlines() if line.strip()]


def check(commits: list[Commit], text: str, scope: str) -> int:
    handles, names = listed(text)
    if not handles:
        print("contributors: CONTRIBUTORS.md lists nobody as `- Name (@handle)` or `- @handle`")
        return 1
    missing = unlisted(commits, handles, names)
    if not missing:
        print(f"contributors: every author of {len(commits)} commit(s) in {scope} is listed")
        return 0
    print(f"contributors: {len(missing)} author(s) of {scope} not listed in CONTRIBUTORS.md:")
    for commit in missing:
        who = f"@{commit.login}" if commit.login else "no GitHub account"
        print(f"  {commit.name} ({who}), e.g. commit {commit.sha[:12]}")
    print(
        "\nAdd each as `- Name (@handle)` or `- @handle` to CONTRIBUTORS.md, in family-name order,"
        "\nas CONTRIBUTING.md's Credit section says. Where GitHub matched a commit to"
        "\nno account, list the person under the name on that commit, or ask them to"
        "\nadd its email to their GitHub account."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("check", help="check commit authors against CONTRIBUTORS.md")
    which = run.add_mutually_exclusive_group(required=True)
    which.add_argument("--pr", type=int, help="the commits of this pull request")
    which.add_argument("--all", action="store_true", help="every commit on the default branch")
    run.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO),
        help="owner/name (default: $GITHUB_REPOSITORY or %(default)s)",
    )
    args = parser.parse_args(argv)

    if args.pr is not None:
        endpoint = f"repos/{args.repo}/pulls/{args.pr}/commits?per_page=100"
        scope = f"pull request #{args.pr}"
    else:
        endpoint = f"repos/{args.repo}/commits?per_page=100"
        scope = "the default branch"
    commits = _gh_commits(endpoint)
    if args.pr is not None and len(commits) >= PR_COMMITS_LIMIT:
        print(
            f"contributors: warning: GitHub returns at most {PR_COMMITS_LIMIT} commits of a"
            " pull request, and this one has that many: any after them were not checked."
        )
    return check(commits, CONTRIBUTORS.read_text(encoding="utf-8"), scope)


if __name__ == "__main__":
    sys.exit(main())
