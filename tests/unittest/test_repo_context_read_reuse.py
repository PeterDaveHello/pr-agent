from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from pr_agent.algo import repo_context


@pytest.fixture(autouse=True)
def context_config(monkeypatch):
    config = {"repo_context_files": [], "repo_context_max_lines": 500}
    monkeypatch.setattr(repo_context, "get_settings", lambda: SimpleNamespace(config=config))
    return config


@pytest.fixture
def provider():
    class FakeProvider:
        get_repo_file_content = Mock(return_value="rules")
        get_repo_context_ref = Mock(return_value="a" * 40)
        list_repo_directory = Mock(return_value=[])
        get_sibling_repo_file_content = Mock(return_value="sibling rules")

    return FakeProvider()


@pytest.mark.parametrize("entries, expect_reuse_tracking", [
    (["AGENTS.md"], False),
    (["AGENTS.md", "CLAUDE.md"], False),
    ([["AGENTS.md", "CLAUDE.md"]], False),
    (["AGENTS.md", ["AGENTS.md", "CLAUDE.md"]], True),
])
def test_reuse_tracking_only_needed_across_alternative_groups(entries, expect_reuse_tracking):
    state = repo_context._RepoContextLoadState(entries)
    assert (state.successful_local_paths is not None) is expect_reuse_tracking


@pytest.mark.parametrize("entries", [
    ["AGENTS.md", ["AGENTS.md", "FALLBACK.md"]],
    [["AGENTS.md"], ["AGENTS.md", "FALLBACK.md"]],
    [["AGENTS.md"], "AGENTS.md"],
], ids=["direct-then-alternative", "repeated-alternatives", "alternative-then-direct"])
@pytest.mark.parametrize("content,expected", [("rules", "rules"), (b"rules", "rules"), (" \n", "")])
def test_successful_read_is_reused_without_selecting_fallback(provider, entries, content, expected):
    provider.get_repo_file_content.return_value = content

    files, had_fetch_error = repo_context._load_repo_context_files(provider, entries, from_default_branch=True)

    assert files == [("AGENTS.md", expected)]
    assert had_fetch_error is False
    provider.get_repo_file_content.assert_called_once_with("AGENTS.md", from_default_branch=True)


@pytest.mark.parametrize("entries", [
    [["rules/*.md"], ["rules/*.md", "FALLBACK.md"]],
    ["rules/a.md", ["rules/*.md", "FALLBACK.md"]],
], ids=["repeated-glob", "partly-loaded-glob"])
def test_glob_alternatives_reuse_reads_and_keep_all_unique_matches(provider, entries):
    provider.list_repo_directory.return_value = ["rules/b.md", "rules/a.md"]
    provider.get_repo_file_content.side_effect = lambda path, **kwargs: path

    files, had_fetch_error = repo_context._load_repo_context_files(provider, entries, from_default_branch=True)

    assert files == [("rules/a.md", "rules/a.md"), ("rules/b.md", "rules/b.md")]
    assert had_fetch_error is False
    assert provider.get_repo_file_content.call_args_list == [
        call("rules/a.md", from_default_branch=True),
        call("rules/b.md", from_default_branch=True),
    ]


@pytest.mark.parametrize("first_read", [None, "", RuntimeError("temporary failure")], ids=["missing", "empty", "error"])
def test_unsuccessful_reads_can_be_retried_by_later_alternatives(provider, first_read):
    provider.get_repo_file_content.side_effect = [first_read, "rules"]

    files, had_fetch_error = repo_context._load_repo_context_files(
        provider, [["AGENTS.md"], ["AGENTS.md", "FALLBACK.md"]], from_default_branch=True
    )

    assert files == [("AGENTS.md", "rules")]
    assert had_fetch_error is isinstance(first_read, Exception)
    assert provider.get_repo_file_content.call_args_list == [
        call("AGENTS.md", from_default_branch=True),
        call("AGENTS.md", from_default_branch=True),
    ]


def test_successful_reads_do_not_leak_into_a_later_build(provider, context_config):
    context_config["repo_context_files"] = ["AGENTS.md", ["AGENTS.md", "FALLBACK.md"]]
    first_context = repo_context.build_repo_context(provider)

    provider.get_repo_file_content.return_value = "updated rules"
    provider.get_repo_context_ref.return_value = "b" * 40
    second_context = repo_context.build_repo_context(provider)

    assert "updated rules" not in first_context
    assert "updated rules" in second_context
    assert provider.get_repo_file_content.call_args_list == [
        call("AGENTS.md", from_default_branch=True),
        call("AGENTS.md", from_default_branch=True),
    ]


@pytest.mark.parametrize("sibling_first", [False, True])
def test_local_read_reuse_does_not_merge_a_sibling_with_the_same_label(provider, sibling_first):
    local_path = "org/shared/AGENTS.md"
    sibling = {"repo_id": "org/shared", "file_path": "AGENTS.md"}
    entries = [sibling, local_path] if sibling_first else [local_path, sibling]
    entries.append([local_path, "FALLBACK.md"])

    files, had_fetch_error = repo_context._load_repo_context_files(provider, entries, from_default_branch=True)

    contents = ["sibling rules", "rules"] if sibling_first else ["rules", "sibling rules"]
    assert files == [(local_path, content) for content in contents]
    assert had_fetch_error is False
    provider.get_repo_file_content.assert_called_once_with(local_path, from_default_branch=True)
    provider.get_sibling_repo_file_content.assert_called_once_with("org/shared", "AGENTS.md")
