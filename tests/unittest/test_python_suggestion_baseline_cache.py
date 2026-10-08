"""Test reuse of validated Python suggestion baselines."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import pr_agent.tools.pr_code_suggestions as suggestions_module


def _source(filename="app.py", content="def value():\n    return 1\n", complete=True):
    return SimpleNamespace(
        filename=filename, head_file=content, head_file_is_complete=complete
    )


def _tool(*files):
    tool = suggestions_module.PRCodeSuggestions.__new__(suggestions_module.PRCodeSuggestions)
    tool.git_provider = SimpleNamespace(diff_files=list(files))
    return tool


def _validate(tool, filename="app.py", snippet="    return 2", start=2, end=2):
    return tool._validate_python_replacement_syntax(filename, start, end, snippet)


def test_initialized_baseline_cache_attribute_is_reused():
    file = _source()
    tool = _tool(file)
    tool._validated_python_sources = {}

    assert _validate(tool) is True
    assert tool._validated_python_sources["app.py"] is file.head_file


def test_repeated_suggestions_compile_baseline_once_but_every_replacement():
    tool = _tool(_source())

    with patch("pr_agent.tools.pr_code_suggestions.compile", wraps=compile, create=True) as compiled:
        assert _validate(tool) is True
        assert _validate(tool, snippet="    return (") is False
        assert _validate(tool, snippet="    return 3") is True

    assert compiled.call_count == 4
    assert [call.args[0] for call in compiled.call_args_list].count(
        "def value():\n    return 1\n"
    ) == 1


def test_changed_file_content_requires_new_baseline_check():
    file = _source()
    tool = _tool(file)
    with patch("pr_agent.tools.pr_code_suggestions.compile", wraps=compile, create=True) as compiled:
        assert _validate(tool) is True
        file.head_file = "def value():\n    return 10\n"
        assert _validate(tool) is True

    assert compiled.call_count == 4


def test_different_files_with_same_source_are_checked_separately():
    content = "def value():\n    return 1\n"
    tool = _tool(_source(filename="first.py", content=content),
                 _source(filename="second.py", content=content))
    with patch("pr_agent.tools.pr_code_suggestions.compile", wraps=compile, create=True) as compiled:
        assert _validate(tool, filename="first.py") is True
        assert _validate(tool, filename="second.py") is True

    assert compiled.call_count == 4


def test_cached_baseline_does_not_cross_tool_instances():
    file = _source()
    with patch("pr_agent.tools.pr_code_suggestions.compile", wraps=compile, create=True) as compiled:
        assert _validate(_tool(file)) is True
        assert _validate(_tool(file)) is True

    assert compiled.call_count == 4


def test_invalid_baseline_is_not_cached():
    tool = _tool(_source(content="def broken(:\n    return 1\n"))
    with patch("pr_agent.tools.pr_code_suggestions.compile", wraps=compile, create=True) as compiled:
        assert _validate(tool) is None
        assert _validate(tool) is None

    assert compiled.call_count == 2


def test_incomplete_file_is_not_checked_or_cached():
    file = _source(complete=False)
    tool = _tool(file)
    with patch("pr_agent.tools.pr_code_suggestions.compile", wraps=compile, create=True) as compiled:
        assert _validate(tool) is None
        file.head_file_is_complete = True
        assert _validate(tool) is True

    assert compiled.call_count == 2


def test_invalid_range_keeps_original_behavior_and_reuses_checked_source():
    tool = _tool(_source())
    with patch("pr_agent.tools.pr_code_suggestions.compile", wraps=compile, create=True) as compiled:
        assert _validate(tool, start=4, end=4) is None
        assert _validate(tool) is True

    assert compiled.call_count == 2


def test_baseline_compile_exception_is_not_cached(monkeypatch):
    original_compile = compile
    calls = []

    def flaky_compile(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            raise RecursionError("baseline too deeply nested")
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(suggestions_module, "compile", flaky_compile, raising=False)
    tool = _tool(_source())

    assert _validate(tool) is None
    assert _validate(tool) is True
    assert len(calls) == 3


@pytest.mark.parametrize("filename", ["app.js", "README.md"])
def test_non_python_files_are_unaffected(filename):
    tool = _tool(_source(filename=filename))
    with patch("pr_agent.tools.pr_code_suggestions.compile", wraps=compile, create=True) as compiled:
        assert _validate(tool, filename=filename) is None

    compiled.assert_not_called()
