import argparse
import json
from pathlib import Path

from .export import replay_bundle


def main():
    parser = argparse.ArgumentParser(description="Verify hashes and rerun an XAS bundle with local Larch.")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        _, _, report = replay_bundle(args.bundle.read_bytes())
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    except (ValueError, OSError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
