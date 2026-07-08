"""Entrypoint — dispatches to offline | online | generate-dataset modes."""

import os
import sys


def main() -> None:
    mode = os.environ.get("EVAL_MODE", "offline").strip().lower()

    if mode == "offline":
        from rag_eval.offline import run_offline_eval
        run_offline_eval()
    elif mode == "online":
        from rag_eval.online_sampler import run_online_eval
        run_online_eval()
    elif mode == "generate-dataset":
        from rag_eval.generate_dataset import generate_dataset
        generate_dataset()
    else:
        print(f"Unknown EVAL_MODE: {mode!r}. Use offline | online | generate-dataset", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
