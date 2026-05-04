"""Tests for cli.py — StatsPAI CLI entry point.

Covers
------
* ``_make_parser`` — argument parser structure (subcommands, flags).
* ``main`` — dispatch logic for each subcommand, return codes, error
  handling, JSON vs text output.
"""

from unittest.mock import MagicMock, patch

import pytest

import statspai
from statspai.cli import _make_parser, main


# ======================================================================
#  _make_parser
# ======================================================================


class TestMakeParser:
    def test_top_level_help(self):
        parser = _make_parser()
        assert parser.prog == "statspai"

    def test_version_flag(self):
        parser = _make_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_version_flag_short(self):
        parser = _make_parser()
        args = parser.parse_args(["-V"])
        assert args.version is True

    def test_list_subcommand(self):
        parser = _make_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"
        assert args.category is None
        assert args.json is False

    def test_list_with_category(self):
        parser = _make_parser()
        args = parser.parse_args(["list", "--category", "causal"])
        assert args.command == "list"
        assert args.category == "causal"

    def test_list_with_json(self):
        parser = _make_parser()
        args = parser.parse_args(["list", "--json"])
        assert args.json is True

    def test_describe_subcommand(self):
        parser = _make_parser()
        args = parser.parse_args(["describe", "did"])
        assert args.command == "describe"
        assert args.name == "did"

    def test_describe_with_json(self):
        parser = _make_parser()
        args = parser.parse_args(["describe", "iv", "--json"])
        assert args.json is True

    def test_search_subcommand(self):
        parser = _make_parser()
        args = parser.parse_args(["search", "causal", "inference"])
        assert args.command == "search"
        assert args.query == ["causal", "inference"]

    def test_search_with_json(self):
        parser = _make_parser()
        args = parser.parse_args(["search", "did", "--json"])
        assert args.json is True

    def test_help_subcommand_no_topic(self):
        parser = _make_parser()
        args = parser.parse_args(["help"])
        assert args.command == "help"
        assert args.topic is None

    def test_help_subcommand_with_topic(self):
        parser = _make_parser()
        args = parser.parse_args(["help", "did"])
        assert args.command == "help"
        assert args.topic == "did"

    def test_help_with_verbose(self):
        parser = _make_parser()
        args = parser.parse_args(["help", "iv", "--verbose"])
        assert args.verbose is True

    def test_help_short_verbose(self):
        parser = _make_parser()
        args = parser.parse_args(["help", "iv", "-v"])
        assert args.verbose is True

    def test_version_subcommand(self):
        parser = _make_parser()
        args = parser.parse_args(["version"])
        assert args.command == "version"

    def test_no_args(self):
        """No args → command is None."""
        parser = _make_parser()
        args = parser.parse_args([])
        assert args.command is None
        assert args.version is False


# ======================================================================
#  main
# ======================================================================


class TestMain:
    def test_version(self, capsys):
        with patch.object(statspai, "__version__", "1.2.3", create=True):
            rc = main(["--version"])
            assert rc == 0
            out, _ = capsys.readouterr()
            assert "1.2.3" in out

    def test_version_subcommand(self, capsys):
        with patch.object(statspai, "__version__", "1.2.3", create=True):
            rc = main(["version"])
            assert rc == 0
            out, _ = capsys.readouterr()
            assert "1.2.3" in out

    def test_no_args_shows_help(self):
        with patch("statspai.help", return_value="overview text") as mock_help:
            rc = main([])
            assert rc == 0
            mock_help.assert_called_once_with()

    def test_list_text(self, capsys):
        with patch("statspai.list_functions", return_value=["did", "iv", "rd"]):
            rc = main(["list"])
            assert rc == 0
            out, _ = capsys.readouterr()
            assert "did" in out
            assert "iv" in out
            assert "rd" in out

    def test_list_json(self, capsys):
        with patch("statspai.list_functions", return_value=["did", "iv"]):
            rc = main(["list", "--json"])
            assert rc == 0
            out, _ = capsys.readouterr()
            import json
            parsed = json.loads(out)
            assert parsed == ["did", "iv"]

    def test_list_empty_text(self, capsys):
        with patch("statspai.list_functions", return_value=[]):
            rc = main(["list", "--category", "bogus"])
            assert rc == 0
            out, _ = capsys.readouterr()
            assert "bogus" in out

    def test_describe_text(self, capsys):
        with patch("statspai.describe_function", return_value={"name": "did"}), \
             patch("statspai.help") as mock_help:
            rc = main(["describe", "did"])
            assert rc == 0
            mock_help.assert_called_once_with("did", verbose=True)

    def test_describe_json(self, capsys):
        with patch("statspai.describe_function",
                   return_value={"name": "did", "category": "causal"}):
            rc = main(["describe", "did", "--json"])
            assert rc == 0
            out, _ = capsys.readouterr()
            import json
            parsed = json.loads(out)
            assert parsed["name"] == "did"

    def test_describe_not_found(self, capsys):
        with patch("statspai.describe_function",
                   side_effect=KeyError("unknown: 'bogus'")):
            rc = main(["describe", "bogus"])
            assert rc == 2
            err = capsys.readouterr().err
            assert "unknown" in err

    def test_search_text(self):
        with patch("statspai.help") as mock_help:
            rc = main(["search", "causal", "inference"])
            assert rc == 0
            mock_help.assert_called_once()

    def test_search_json(self, capsys):
        with patch("statspai.search_functions", return_value=["did", "iv"]):
            rc = main(["search", "did", "--json"])
            assert rc == 0
            out, _ = capsys.readouterr()
            import json
            parsed = json.loads(out)
            assert parsed == ["did", "iv"]

    def test_help_no_topic(self):
        with patch("statspai.help") as mock_help:
            rc = main(["help"])
            assert rc == 0
            mock_help.assert_called_once_with()

    def test_help_with_topic(self):
        with patch("statspai.help") as mock_help:
            rc = main(["help", "did"])
            assert rc == 0
            mock_help.assert_called_once_with("did", verbose=False)

    def test_help_verbose(self):
        with patch("statspai.help") as mock_help:
            rc = main(["help", "did", "--verbose"])
            assert rc == 0
            mock_help.assert_called_once_with("did", verbose=True)

    def test_unknown_command(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["bogus"])
        assert exc.value.code == 2
