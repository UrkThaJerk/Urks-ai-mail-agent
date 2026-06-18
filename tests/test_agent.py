"""Basic test suite for the AI mail agent project."""

import importlib
import os
import sys

import pytest

# Ensure repository root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports:
    """Verify that the main agent modules can be imported without errors."""

    def test_agent_module_importable(self):
        """agent.py should be locatable on sys.path.

        We use find_spec rather than import_module because the modules execute
        top-level code (OpenAI client init, dotenv load) that requires
        credentials not available in CI.
        """
        spec = importlib.util.find_spec("agent")
        assert spec is not None, "agent module not found on sys.path"

    def test_mail_agent_module_importable(self):
        """mail_agent.py should be locatable on sys.path.

        We use find_spec rather than import_module because the modules execute
        top-level code (OpenAI client init, dotenv load) that requires
        credentials not available in CI.
        """
        spec = importlib.util.find_spec("mail_agent")
        assert spec is not None, "mail_agent module not found on sys.path"


class TestDependencies:
    """Verify that required third-party packages are installed."""

    def test_langchain_available(self):
        """langchain package should be installed."""
        langchain = importlib.util.find_spec("langchain")
        assert langchain is not None, "langchain is not installed"

    def test_langgraph_available(self):
        """langgraph package should be installed."""
        langgraph = importlib.util.find_spec("langgraph")
        assert langgraph is not None, "langgraph is not installed"

    def test_openai_available(self):
        """openai package should be installed."""
        openai = importlib.util.find_spec("openai")
        assert openai is not None, "openai is not installed"


class TestProjectStructure:
    """Validate that expected project files are present and non-empty."""

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_requirements_txt_exists(self):
        """requirements.txt must exist in the repository root."""
        path = os.path.join(self.REPO_ROOT, "requirements.txt")
        assert os.path.isfile(path), "requirements.txt not found"

    def test_requirements_txt_has_content(self):
        """requirements.txt must not be empty."""
        path = os.path.join(self.REPO_ROOT, "requirements.txt")
        assert os.path.getsize(path) > 0, "requirements.txt is empty"

    def test_agent_py_exists(self):
        """agent.py must exist in the repository root."""
        path = os.path.join(self.REPO_ROOT, "agent.py")
        assert os.path.isfile(path), "agent.py not found"

    def test_mail_agent_py_exists(self):
        """mail_agent.py must exist in the repository root."""
        path = os.path.join(self.REPO_ROOT, "mail_agent.py")
        assert os.path.isfile(path), "mail_agent.py not found"
