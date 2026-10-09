"""Export a specified Git commit into a new candidate snapshot directory."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from prepare_source import extract_archive, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True, help="full 40-character upstream commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise ValueError("pin an exact full commit, not a moving branch or tag")
    resolved = subprocess.check_output(["git", "rev-parse", f"{args.commit}^{{commit}}"], cwd=args.repo, text=True).strip()
    if resolved != args.commit:
        raise ValueError("commit identity mismatch")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("candidate output must be new")
    output.mkdir(parents=True)
    archive = output / f"freetoken-{args.commit[:7]}.zip"
    subprocess.run(["git", "archive", "--format=zip", "--output", str(archive), args.commit], cwd=args.repo, check=True)
    extract_archive(archive, output / "FreeToken")
    (output / "candidate-lock.json").write_text(json.dumps({
        "official_repository": "https://github.com/FlashML-org/FreeToken",
        "commit": args.commit, "archive": archive.name, "archive_sha256": sha256(archive),
        "browse_tree": "FreeToken", "snapshot_modified": False, "license": "Apache-2.0",
        "status": "candidate_requires_patch_review_and_validation"
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {args.commit} to {output}; existing profiles were not changed.")


if __name__ == "__main__":
    main()
