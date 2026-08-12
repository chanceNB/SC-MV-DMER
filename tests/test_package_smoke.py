from sc_mv_dmer import __version__, app_name


def test_package_exposes_project_identity() -> None:
    assert __version__ == "0.1.0"
    assert app_name() == "sc-mv-dmer"
