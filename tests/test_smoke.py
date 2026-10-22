"""Smoke tests that do not call the Groq API."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from multi_agent import (
    ConfigurationError,
    RouteDecision,
    build_team,
    list_knowledge_files,
    require_api_key,
    route_supervisor,
)


class ConfigTests(unittest.TestCase):
    def test_require_api_key_rejects_missing(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            with self.assertRaises(ConfigurationError):
                require_api_key()

    def test_require_api_key_rejects_placeholder(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_your-key-here"}, clear=False):
            with self.assertRaises(ConfigurationError):
                require_api_key()

    def test_require_api_key_accepts_value(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_real_looking"}, clear=False):
            self.assertEqual(require_api_key(), "gsk_test_real_looking")


class KnowledgeTests(unittest.TestCase):
    def test_knowledge_corpus_present(self) -> None:
        files = list_knowledge_files()
        self.assertGreaterEqual(len(files), 4)
        names = {p.name for p in files}
        self.assertIn("onboarding_faq.md", names)
        self.assertIn("platform_notes.md", names)


class GraphTests(unittest.TestCase):
    def test_build_team_compiles(self) -> None:
        build_team.cache_clear()
        app = build_team()
        self.assertIsNotNone(app)

    def test_route_supervisor_reads_state(self) -> None:
        self.assertEqual(
            route_supervisor(
                {
                    "question": "q",
                    "route": "analyst",
                    "research_notes": "",
                    "analysis": "",
                    "final_answer": "",
                    "steps": [],
                    "hop_count": 0,
                }
            ),
            "analyst",
        )

    def test_route_decision_schema(self) -> None:
        decision = RouteDecision(next_agent="writer", reason="analysis ready")
        self.assertEqual(decision.next_agent, "writer")


if __name__ == "__main__":
    unittest.main()
