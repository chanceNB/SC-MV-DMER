from sc_mv_dmer import __version__, app_name
from sc_mv_dmer.cli import main


def test_package_exposes_project_identity() -> None:
    assert __version__ == "0.1.0"
    assert app_name() == "sc-mv-dmer"


def test_cli_entry_prints_project_name(capsys) -> None:
    assert main() == 0
    assert capsys.readouterr().out == "sc-mv-dmer\n"
