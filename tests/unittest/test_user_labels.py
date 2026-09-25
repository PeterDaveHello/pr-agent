from unittest.mock import Mock

import pytest

from pr_agent.algo import utils
from pr_agent.algo.output_models import Label, PRType


@pytest.fixture
def settings(monkeypatch):
    settings = Mock()
    settings.config.get.return_value = False
    settings.get.return_value = {}
    monkeypatch.setattr(utils, "get_settings", lambda: settings)
    return settings


@pytest.mark.parametrize("label", ["Bug fix with tests", "bug fix with tests", "BUG FIX WITH TESTS"])
@pytest.mark.parametrize("enabled", [False, True])
def test_unadvertised_label_is_preserved(settings, label, enabled):
    settings.config.get.return_value = enabled
    current = [label, "team:backend", "release-blocker"]

    assert utils.get_user_labels(current) == current
    assert current == [label, "team:backend", "release-blocker"]


@pytest.mark.parametrize("label_enum", [PRType, Label])
@pytest.mark.parametrize("enabled", [False, True])
def test_shipped_classifications_are_excluded(settings, label_enum, enabled):
    settings.config.get.return_value = enabled
    labels = [member.value.upper() for member in label_enum]
    current = ["team:backend", *labels, "Bug fix with tests", "release-blocker"]

    assert utils.get_user_labels(current) == ["team:backend", "Bug fix with tests", "release-blocker"]
    assert current == ["team:backend", *labels, "Bug fix with tests", "release-blocker"]


@pytest.mark.parametrize("label", ["Bug fix with tests", "Custom category"])
@pytest.mark.parametrize("enabled", [False, True])
def test_custom_labels_are_excluded_only_when_enabled(settings, label, enabled):
    settings.config.get.return_value = enabled
    settings.get.return_value = {label: {"description": "A custom category"}}
    current = [label, "team:backend"]
    expected = ["team:backend"] if enabled else current

    assert utils.get_user_labels(current) == expected
    assert current == [label, "team:backend"]


@pytest.mark.parametrize("current", [None, []])
def test_empty_labels(settings, current):
    assert utils.get_user_labels(current) == []
