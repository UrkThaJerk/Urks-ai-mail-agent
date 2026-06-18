"""
Basic tests for the AI Mail Agent
These tests serve as a foundation for continuous testing
and help other agents learn about the codebase structure.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAgentImports:
    """Test that core modules can be imported successfully"""

    def test_agent_module_imports(self):
        """Verify agent.py can be imported"""
        try:
            import agent
            assert agent is not None
        except ImportError as e:
            pytest.fail(f"Failed to import agent module: {e}")

    def test_mail_agent_module_imports(self):
        """Verify mail_agent.py can be imported"""
        try:
            import mail_agent
            assert mail_agent is not None
        except ImportError as e:
            pytest.fail(f"Failed to import mail_agent module: {e}")


class TestDependencies:
    """Test that required dependencies are available"""

    def test_langchain_available(self):
        """Verify langchain is installed"""
        try:
            import langchain
            assert langchain is not None
        except ImportError:
            pytest.fail("langchain is not installed")

    def test_langgraph_available(self):
        """Verify langgraph is installed"""
        try:
            import langgraph
            assert langgraph is not None
        except ImportError:
            pytest.fail("langgraph is not installed")

    def test_openai_available(self):
        """Verify openai is installed"""
        try:
            import openai
            assert openai is not None
        except ImportError:
            pytest.fail("openai is not installed")


class TestBasicStructure:
    """Test basic code structure and patterns"""

    def test_requirements_file_exists(self):
        """Verify requirements.txt exists"""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        assert req_file.exists(), "requirements.txt not found"

    def test_requirements_file_not_empty(self):
        """Verify requirements.txt has content"""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        with open(req_file) as f:
            content = f.read().strip()
            assert len(content) > 0, "requirements.txt is empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
