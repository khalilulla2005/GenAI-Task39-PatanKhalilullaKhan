from __future__ import annotations

import argparse
import sys

from src.assistant import check_ollama, create_chat_model, run_assistant
from src.config import Settings
from src.prompts import TaskType


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask a local Ollama CodeLlama model for coding help."
    )
    parser.add_argument(
        "--task",
        choices=[task.value for task in TaskType],
        default=TaskType.GENERATE.value,
        help="Coding task to perform (default: Generate Code).",
    )
    parser.add_argument(
        "--prompt",
        help="Question or source code to submit. Required unless checking the connection.",
    )
    parser.add_argument(
        "--check-connection",
        action="store_true",
        help="Check Ollama availability and the configured model, then exit.",
    )
    parser.add_argument(
        "--base-url",
        help="Override OLLAMA_BASE_URL for this run.",
    )
    parser.add_argument(
        "--model",
        help="Override OLLAMA_MODEL for this run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.base_url or args.model:
            settings = settings.with_overrides(
                base_url=args.base_url,
                model=args.model,
            )

        if args.check_connection:
            status = check_ollama(settings)
            print(f"Ollama is running at {settings.base_url}.")
            if status.model_available:
                print(f"Model {settings.model!r} is available.")
                return 0
            print(
                f"Model {settings.model!r} is not installed. Run "
                f"`ollama pull {settings.model}`.",
                file=sys.stderr,
            )
            if status.installed_models:
                print(
                    "Installed models: " + ", ".join(status.installed_models),
                    file=sys.stderr,
                )
            return 2

        if not args.prompt or not args.prompt.strip():
            parser.error("--prompt is required unless --check-connection is used.")

        status = check_ollama(settings)
        if not status.model_available:
            print(
                f"Model {settings.model!r} is not installed. Run "
                f"`ollama pull {settings.model}`.",
                file=sys.stderr,
            )
            return 2

        answer = run_assistant(
            args.task,
            args.prompt,
            settings,
            model=create_chat_model(settings),
        )
        print(answer)
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
