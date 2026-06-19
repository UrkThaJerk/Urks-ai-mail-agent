import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from collective_agent import CollectiveLearningAgent, process_collective_jobs


class FakeCompletions:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses[len(self.calls) - 1]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class CollectiveAgentTests(unittest.TestCase):
    def test_run_collective_collects_independent_work_peer_learning_and_summary(self):
        client = FakeClient(
            [
                "mail independent",
                "video independent",
                "social independent",
                "strategy independent",
                "mail learned from peers",
                "video learned from peers",
                "social learned from peers",
                "strategy learned from peers",
                "collective summary",
            ]
        )

        result = CollectiveLearningAgent(client=client).run_collective(
            "Improve the workflows",
            "Keep the four agents separate before they compare notes.",
        )

        self.assertEqual("Improve the workflows", result["objective"])
        self.assertEqual(4, len(result["agents"]))
        self.assertEqual("mail independent", result["agents"][0]["independent_work"])
        self.assertEqual("strategy learned from peers", result["agents"][3]["peer_learning"])
        self.assertEqual("collective summary", result["collective_summary"])
        self.assertEqual(9, len(client.chat.completions.calls))

    @patch("collective_agent.get_openai_client", return_value=object())
    @patch("collective_agent.CollectiveLearningAgent.run_collective")
    @patch("builtins.print")
    def test_process_collective_jobs_writes_output_file_when_requested(
        self,
        print_mock,
        run_collective,
        _get_openai_client,
    ):
        run_collective.return_value = {"objective": "Improve", "agents": [], "collective_summary": "Done"}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir, "collective.json")
            with patch.dict(
                "os.environ",
                {
                    "COLLECTIVE_OBJECTIVE": "Improve",
                    "COLLECTIVE_CONTEXT": "Shared context",
                    "COLLECTIVE_OUTPUT_PATH": str(output_path),
                },
                clear=False,
            ):
                process_collective_jobs()

            self.assertTrue(output_path.exists())
            self.assertEqual(run_collective.return_value, json.loads(output_path.read_text(encoding="utf-8")))
            print_mock.assert_called_once()

    def test_process_collective_jobs_requires_objective(self):
        with patch.dict("os.environ", {"COLLECTIVE_OBJECTIVE": ""}, clear=False):
            with self.assertRaisesRegex(ValueError, "COLLECTIVE_OBJECTIVE"):
                process_collective_jobs()


if __name__ == "__main__":
    unittest.main()
