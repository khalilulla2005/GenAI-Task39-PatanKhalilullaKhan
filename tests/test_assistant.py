from __future__ import annotations

import json
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.request import Request

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.assistant import check_ollama, create_chat_model, run_assistant
from src.config import Settings
from src.prompts import TASK_INSTRUCTIONS, TaskType, build_chat_prompt


def _settings(**overrides: str) -> Settings:
    values = {"OLLAMA_MODEL": "codellama:7b"}
    values.update(overrides)
    return Settings.from_env(values)


class FakeChatModel(BaseChatModel):
    seen_messages: list = []

    @property
    def _llm_type(self) -> str:
        return "fake-codellama-test-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages = messages
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="def is_prime(number: int) -> bool:\n    return True"
                    )
                )
            ]
        )


class SettingsTests(unittest.TestCase):
    def test_defaults_target_local_codellama_7b(self):
        settings = Settings.from_env({})

        self.assertEqual(settings.base_url, "http://127.0.0.1:11434")
        self.assertEqual(settings.model, "codellama:7b")
        self.assertEqual(settings.temperature, 0.2)

    def test_streamlit_secrets_supply_cloud_settings_without_exposing_key(self):
        settings = Settings.from_env(
            {},
            secrets={
                "OLLAMA_BASE_URL": "https://ollama.example.com",
                "OLLAMA_MODEL": "codellama:7b",
                "OLLAMA_API_KEY": "private-token",
            },
        )

        self.assertEqual(settings.base_url, "https://ollama.example.com")
        self.assertNotIn("private-token", repr(settings))

    def test_environment_overrides_streamlit_secrets(self):
        settings = Settings.from_env(
            {"OLLAMA_MODEL": "codellama:13b"},
            secrets={"OLLAMA_MODEL": "codellama:7b"},
        )

        self.assertEqual(settings.model, "codellama:13b")

    def test_settings_reject_invalid_url_model_and_generation_values(self):
        invalid_values = (
            ("OLLAMA_BASE_URL", "https://user:password@example.com"),
            ("OLLAMA_BASE_URL", "file:///tmp/ollama"),
            ("OLLAMA_MODEL", "codellama:7b?unexpected=true"),
            ("OLLAMA_TEMPERATURE", "2.1"),
            ("OLLAMA_NUM_CTX", "128"),
            ("OLLAMA_REQUEST_TIMEOUT", "0"),
        )
        for name, value in invalid_values:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    _settings(**{name: value})

    def test_overrides_are_validated_and_preserve_other_settings(self):
        settings = _settings().with_overrides(model="qwen2.5-coder:7b")
        self.assertEqual(settings.model, "qwen2.5-coder:7b")
        self.assertEqual(settings.base_url, "http://127.0.0.1:11434")
        with self.assertRaisesRegex(ValueError, "OLLAMA_MODEL"):
            _settings().with_overrides(model="invalid model")


class PromptTests(unittest.TestCase):
    def test_every_task_has_clear_task_specific_output_instructions(self):
        self.assertEqual(set(TASK_INSTRUCTIONS), set(TaskType))
        for task, instruction in TASK_INSTRUCTIONS.items():
            with self.subTest(task=task):
                prompt = build_chat_prompt(task).format_messages(
                    request="sample code"
                )
                self.assertIn(task.value, prompt[1].content)
                self.assertIn(instruction, prompt[1].content)
                self.assertIn("sample code", prompt[1].content)
                self.assertIn("does not execute", prompt[0].content)

    def test_unknown_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            build_chat_prompt("Review Code")


class AssistantTests(unittest.TestCase):
    def test_request_uses_task_prompt_and_returns_model_response(self):
        model = FakeChatModel()
        answer = run_assistant(
            TaskType.GENERATE,
            "  Write a prime checker.  ",
            _settings(),
            model=model,
        )

        self.assertIn("def is_prime", answer)
        prompt_text = str(model.seen_messages[-1].content)
        self.assertIn("Write a prime checker.", prompt_text)
        self.assertIn("complete, runnable Python 3 code", prompt_text)

    def test_empty_request_and_empty_model_response_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "request"):
            run_assistant(TaskType.GENERATE, "  ", _settings(), model=FakeChatModel())
        with self.assertRaisesRegex(ValueError, "character limit"):
            run_assistant(
                TaskType.GENERATE,
                "x" * 20001,
                _settings(),
                model=FakeChatModel(),
            )

        class EmptyModel(FakeChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="  "))]
                )

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            run_assistant(
                TaskType.GENERATE,
                "Make a function.",
                _settings(),
                model=EmptyModel(),
            )

    def test_model_factory_passes_model_settings_and_optional_gateway_token(self):
        created = {}

        class FakeOllama:
            def __init__(self, **kwargs):
                created.update(kwargs)

        fake_settings = _settings(OLLAMA_API_KEY="test-token")
        with patch("langchain_ollama.ChatOllama", FakeOllama):
            result = create_chat_model(fake_settings)

        self.assertIsInstance(result, FakeOllama)
        self.assertEqual(created["model"], "codellama:7b")
        self.assertEqual(created["base_url"], fake_settings.base_url)
        self.assertEqual(
            created["client_kwargs"]["headers"]["Authorization"],
            "Bearer test-token",
        )

    def test_connection_check_reports_model_presence_and_gateway_header(self):
        response_body = json.dumps(
            {"models": [{"name": "codellama:7b"}, {"name": "llama3:8b"}]}
        ).encode()
        captured = {}

        class FakeResponse(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        def fake_urlopen(request: Request, timeout: int):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse(response_body)

        settings = _settings(OLLAMA_API_KEY="gateway-token")
        with patch("src.assistant.urlopen", side_effect=fake_urlopen):
            status = check_ollama(settings)

        self.assertTrue(status.available)
        self.assertTrue(status.model_available)
        self.assertEqual(status.installed_models, ("codellama:7b", "llama3:8b"))
        self.assertEqual(
            captured["request"].get_header("Authorization"),
            "Bearer gateway-token",
        )
        self.assertEqual(captured["timeout"], settings.request_timeout)

    def test_connection_check_distinguishes_missing_model_and_unreachable_service(self):
        response_body = json.dumps({"models": [{"name": "llama3:8b"}]}).encode()

        class FakeResponse(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        with patch(
            "src.assistant.urlopen",
            return_value=FakeResponse(response_body),
        ):
            status = check_ollama(_settings())
        self.assertTrue(status.available)
        self.assertFalse(status.model_available)

        with patch("src.assistant.urlopen", side_effect=OSError("connection refused")):
            with self.assertRaisesRegex(RuntimeError, "Could not reach Ollama"):
                check_ollama(_settings())


if __name__ == "__main__":
    unittest.main()
