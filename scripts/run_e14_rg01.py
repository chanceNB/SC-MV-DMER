"""Execute the E1.4 RG-01 formal closure evaluation."""

from pathlib import Path

from sc_mv_dmer.gates.rg01 import RG01Evidence, write_rg01_evidence


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    write_rg01_evidence(
        RG01Evidence(root),
        attempt_path=root / "evidence/gates/rg-01/rg-01-attempt-v1.json",
        qualification_path=root / "evidence/qualifications/e1-4-rg01-formal-closure.json",
        report_path=root / "reports/experiments/e1-4-rg01-formal-closure.json",
        text_report_path=root / "reports/experiments/e1-4-rg01-formal-closure-report.txt",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
