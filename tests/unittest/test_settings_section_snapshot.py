import copy

import pytest
from dynaconf.base import UPPER_DEFAULT_SETTINGS
from starlette_context import context, request_cycle_context

from pr_agent.config_loader import get_settings, global_settings
from pr_agent.git_providers import utils as git_utils


@pytest.fixture
def request_settings():
    with request_cycle_context({}):
        context["settings"] = copy.deepcopy(global_settings)
        yield get_settings()


def test_snapshot_preserves_as_dict_section_semantics(request_settings):
    settings = request_settings
    settings.set(
        "PR_REVIEWER",
        {"num_max_findings": 7, "nested": {"items": [{"value": 1}, ["x", "y"]]}},
        merge=False,
    )

    expected = copy.deepcopy(settings.as_dict()["PR_REVIEWER"])
    actual = git_utils._snapshot_settings_section(settings, "pr_reviewer")

    assert actual == expected
    assert type(actual) is dict
    assert type(actual["nested"]) is dict
    assert type(actual["nested"]["items"]) is list
    actual["nested"]["items"][0]["value"] = 99
    assert settings.get("PR_REVIEWER").get("nested")["items"][0]["value"] == 1


def test_snapshot_keeps_as_dict_internal_key_exclusions(request_settings):
    assert "ENV_FOR_DYNACONF" in UPPER_DEFAULT_SETTINGS
    assert git_utils._snapshot_settings_section(request_settings, "ENV_FOR_DYNACONF") == {}
    assert git_utils._snapshot_settings_section(request_settings, "NONEXISTENT_SECTION") == {}


def test_snapshot_does_not_reload_fresh_sections(request_settings):
    settings = request_settings
    settings.set("PR_REVIEWER", {"num_max_findings": 19, "extra_instructions": "global-overlay"}, merge=False)
    settings.__core__.config.fresh_vars.append("PR_REVIEWER")
    expected = copy.deepcopy(settings.as_dict()["PR_REVIEWER"])

    assert git_utils._snapshot_settings_section(settings, "pr_reviewer") == expected
    assert settings.as_dict()["PR_REVIEWER"] == expected


def test_layered_repo_settings_survive_fresh_var_reload_config(tmp_path, request_settings):
    settings = request_settings
    root = tmp_path / "global.toml"
    local = tmp_path / "local.toml"
    root.write_text('[pr_reviewer]\nextra_instructions = "from-global"\n', encoding="utf-8")
    local.write_text("[pr_reviewer]\nnum_max_findings = 13\n", encoding="utf-8")
    settings.__core__.config.fresh_vars.append("PR_REVIEWER")

    git_utils._apply_repo_settings_file(str(root))
    assert settings.as_dict()["PR_REVIEWER"]["extra_instructions"] == "from-global"

    git_utils._apply_repo_settings_file(str(local), repo_settings_scope="per_directory")
    section = settings.as_dict()["PR_REVIEWER"]
    assert section["extra_instructions"] == "from-global"
    assert section["num_max_findings"] == 13


def test_repo_merge_normalizes_only_sections_being_updated(tmp_path, monkeypatch, request_settings):
    path = tmp_path / ".pr_agent.toml"
    path.write_text(
        '[pr_reviewer]\nextra_instructions = "review-only"\n'
        '[pr_code_suggestions]\nextra_instructions = "improve-only"\n',
        encoding="utf-8",
    )
    original_to_dict = git_utils.to_dict
    visited_sections = []

    def track_conversion(section):
        visited_sections.append(section)
        return original_to_dict(section)

    monkeypatch.setattr(git_utils, "to_dict", track_conversion)
    git_utils._apply_repo_settings_file(str(path))

    assert len(visited_sections) == 2
    assert all(isinstance(section, dict) for section in visited_sections)
    assert "CONFIG" not in visited_sections[0]
    assert "CONFIG" not in visited_sections[1]
    assert request_settings.get("PR_REVIEWER.EXTRA_INSTRUCTIONS") == "review-only"
    assert request_settings.get("PR_CODE_SUGGESTIONS.EXTRA_INSTRUCTIONS") == "improve-only"
