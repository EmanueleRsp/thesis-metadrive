from __future__ import annotations

from thesis_rl.cli.scenarios import check_database


def test_check_database_parser_exposes_official_check_choices(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["check_database", "overlap", "/tmp/one", "--other-database-path", "/tmp/two"],
    )

    class _Result:
        returncode = 0
        stdout = "No overlapping in two database!\n"
        stderr = ""

    monkeypatch.setattr(check_database, "run_official_check", lambda *args, **kwargs: _Result())
    assert check_database.main() == 0
    assert "No overlapping" in capsys.readouterr().out
