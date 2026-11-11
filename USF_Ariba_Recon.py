"""USF-Ariba Reconciliation (interim monitoring control)

Compares the Unified Storefront (USF) export with the Ariba Sourcing Request report and lists
USF requests that are missing in Ariba or need investigation.

Read-only: it reads two report files and writes a new results workbook. It never connects to,
creates or changes anything in Ariba or USF.

HOW TO RUN
  Double-click this file (Windows), or run:  python USF_Ariba_Recon.py
  Two file pickers open: choose the USF export, then the Ariba report.
  The results workbook is saved next to the USF export and opens automatically.

  Or pass the files directly:
      python USF_Ariba_Recon.py --usf USF.xlsx --ariba Ariba.xlsx [--no-open]

NEEDS  Python 3.9+ with pandas and openpyxl:  python -m pip install --user pandas openpyxl

FIRST-TIME SETUP
  Run once with the real reports, open the "USF Status Values" tab, then fill in
  USF_STATUS_COLUMN and the status lists in the SETTINGS block below. Until then nothing is
  flagged Missing; unmatched requests show as "Not Found - Eligibility Unconfirmed".

TRACKING
  First-detected-missing and first-observed-in-Ariba dates are kept in a small hidden file,
  ~/.usf_ariba_recon_history.csv (your home folder). Delete it to reset tracking.
"""

# =============================================================================
# SETTINGS: edit the values in quotes. Names are matched ignoring case and extra spaces.
# =============================================================================

USF_ID_COLUMN = "ID"

# Column holding the USF request status. Leave "" until confirmed from a real export.
USF_STATUS_COLUMN = ""

# eligible:     should already have an Ariba SR. Unmatched = Missing.
# pending:      not yet approved. Unmatched = Pending (not an exception).
# not_required: cancelled, rejected, withdrawn. Unmatched = Not Required.
# Any status not listed is reported as "Unknown Status" so new values surface.
ELIGIBLE_STATUSES = []        # for example ["Approved"]
PENDING_STATUSES = []         # for example ["Submitted", "Draft"]
NOT_REQUIRED_STATUSES = []    # for example ["Rejected", "Cancelled", "Withdrawn"]

# Extra USF columns to show in the output, for example ["Title", "Requester"]. Empty = all.
DISPLAY_COLUMNS = []

ARIBA_PROJECT_NAME_COLUMN = "[SREQ]Project(Project Name)"
ARIBA_PROJECT_ID_COLUMN = "[SREQ]Project Id"
ARIBA_STATUS_COLUMN = "Status"

# Ariba statuses that do not count as a valid SR, for example ["Canceled"].
ARIBA_INACTIVE_STATUSES = []

# Identifier prefix embedded in the Ariba project name. 1901 becomes USF_1901.
ID_PREFIX = "USF_"

HISTORY_FILE = "~/.usf_ariba_recon_history.csv"

# =============================================================================
# Code below this line does not need editing.
# =============================================================================

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError:
    print("This tool needs pandas and openpyxl. Install them with:\n"
          "    python -m pip install --user pandas openpyxl")
    if sys.platform == "win32":
        input("Press Enter to close.")
    sys.exit(1)


class ConfigError(Exception):
    pass


class InputError(Exception):
    pass


# Categories, in output sort order (exceptions first)
MISSING = "Missing"
DUPLICATE = "Duplicate"
REVIEW = "Review"
UNKNOWN = "Unknown Status"
UNCONFIRMED = "Not Found - Eligibility Unconfirmed"
CREATED = "Created"
PENDING = "Pending"
NOT_REQUIRED = "Not Required"
CATEGORY_ORDER = [MISSING, DUPLICATE, REVIEW, UNKNOWN, UNCONFIRMED, CREATED, PENDING, NOT_REQUIRED]
ACTION_CATEGORIES = {MISSING, DUPLICATE, REVIEW, UNKNOWN, UNCONFIRMED}

ELIGIBLE, BUCKET_PENDING, BUCKET_NOT_REQUIRED, BUCKET_UNKNOWN, BUCKET_UNCONFIGURED = (
    "Eligible", "Pending", "Not Required", "Unknown", "Unconfigured")

SUPPORTED = {".xlsx", ".xlsm", ".csv"}
HEADER_SCAN_ROWS = 30
_NUMERIC_ID = re.compile(r"^\s*(\d+)(?:\.0+)?\s*$")


def norm(value) -> str:
    """Trimmed, lower case, single spaces. Used for column and status comparisons."""
    if value is None:
        return ""
    return " ".join(str(value).split()).lower()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    usf_id_column: str = USF_ID_COLUMN
    usf_status_column: str = USF_STATUS_COLUMN
    eligible_statuses: set = field(default_factory=lambda: _norm_set(ELIGIBLE_STATUSES))
    pending_statuses: set = field(default_factory=lambda: _norm_set(PENDING_STATUSES))
    not_required_statuses: set = field(default_factory=lambda: _norm_set(NOT_REQUIRED_STATUSES))
    display_columns: list = field(default_factory=lambda: list(DISPLAY_COLUMNS))
    ariba_name_column: str = ARIBA_PROJECT_NAME_COLUMN
    ariba_id_column: str = ARIBA_PROJECT_ID_COLUMN
    ariba_status_column: str = ARIBA_STATUS_COLUMN
    ariba_inactive_statuses: set = field(default_factory=lambda: _norm_set(ARIBA_INACTIVE_STATUSES))
    id_prefix: str = ID_PREFIX
    history_file: Path = field(default_factory=lambda: Path(HISTORY_FILE).expanduser())
    warnings: list = field(default_factory=list)

    @property
    def eligibility_configured(self) -> bool:
        return bool(self.usf_status_column and self.eligible_statuses)

    def validate(self) -> "Config":
        for name in ("eligible_statuses", "pending_statuses", "not_required_statuses", "ariba_inactive_statuses"):
            setattr(self, name, _norm_set(getattr(self, name)))
        e, p, n = self.eligible_statuses, self.pending_statuses, self.not_required_statuses
        overlap = (e & p) | (e & n) | (p & n)
        if overlap:
            raise ConfigError(f"These statuses appear in more than one status list: {sorted(overlap)}")
        if not self.usf_status_column:
            self.warnings.append(
                "USF status column is not set. No request is flagged Missing until USF_STATUS_COLUMN and "
                "ELIGIBLE_STATUSES are set at the top of USF_Ariba_Recon.py.")
        elif not self.eligible_statuses:
            self.warnings.append(
                "ELIGIBLE_STATUSES is empty. No request is flagged Missing until it is set. "
                "See the USF Status Values tab for the values found.")
        return self


def _norm_set(values) -> set:
    return {norm(v) for v in values if norm(v)}


# =============================================================================
# Reading reports (read-only)
# =============================================================================

def _raw_sheets(path: Path) -> dict:
    if path.suffix.lower() not in SUPPORTED:
        raise InputError(f"{path.name}: unsupported file type. Save it as .xlsx or .csv.")
    try:
        if path.suffix.lower() == ".csv":
            return {"csv": pd.read_csv(path, header=None, dtype=str, keep_default_na=False, encoding_errors="replace")}
        return pd.read_excel(path, sheet_name=None, header=None, dtype=str, keep_default_na=False)
    except PermissionError as exc:
        raise InputError(f"Cannot open {path.name}. Close it in Excel and run again.") from exc
    except Exception as exc:
        raise InputError(f"Cannot read {path.name}: {exc}") from exc


def read_report(path: Path, required: list, label: str) -> pd.DataFrame:
    """Read the first sheet whose header row (within the first rows) has every required column."""
    wanted = {norm(c) for c in required if c}
    seen = []
    for sheet, raw in _raw_sheets(path).items():
        for i in range(min(HEADER_SCAN_ROWS, len(raw))):
            row = [str(v).strip() for v in raw.iloc[i].tolist()]
            if wanted <= {norm(v) for v in row}:
                df = raw.iloc[i + 1:].copy()
                df.columns = _unique_names(row)
                df = df[[c for c in df.columns if c]]
                return df[(df != "").any(axis=1)].reset_index(drop=True)
        if len(raw):
            seen.append(f"{sheet}: {[v for v in raw.iloc[0].tolist() if str(v).strip()][:15]}")
    raise InputError(
        f"The {label} report {path.name} does not contain the columns {sorted(c for c in required if c)}.\n"
        f"First row found: {seen}\nCheck the column names in the SETTINGS block of USF_Ariba_Recon.py.")


def resolve_column(df: pd.DataFrame, name: str) -> str:
    for col in df.columns:
        if norm(col) == norm(name):
            return col
    raise InputError(f'Column "{name}" not found. Available columns: {list(df.columns)}')


def _unique_names(row: list) -> list:
    seen, out = {}, []
    for name in row:
        if name and name in seen:
            seen[name] += 1
            out.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


# =============================================================================
# Matching
# =============================================================================

class Extractor:
    """Pulls complete USF identifiers out of free-text Ariba project names."""

    def __init__(self, prefix: str):
        core = prefix.strip("_- ") or prefix
        # Complete identifier: prefix + digits, not glued to letters/digits on either side.
        self.strict = re.compile(rf"(?<![A-Za-z0-9]){re.escape(prefix)}(\d+)(?![A-Za-z0-9])", re.IGNORECASE)
        # Near-misses such as "USF 1901", "USF-1901", "USF1901", "USF_1901A". Never auto-matched.
        self.loose = re.compile(rf"(?<![A-Za-z0-9]){re.escape(core)}\s*[-_#:.]?\s*(\d+)(?!\d)", re.IGNORECASE)
        self.prefix = prefix

    def ids(self, text: str) -> list:
        return list(dict.fromkeys(int(m) for m in self.strict.findall(text or "")))

    def malformed_ids(self, text: str) -> list:
        return list(dict.fromkeys(int(m) for m in self.loose.findall(text or "")))

    def key(self, number: int) -> str:
        return f"{self.prefix}{number}"


def parse_usf_id(value, extractor: Extractor):
    """Numeric USF ID. Accepts 1901, '1901', '1901.0' and 'USF_1901'. Returns None if invalid."""
    text = str(value)
    m = _NUMERIC_ID.match(text)
    if m:
        return int(m.group(1))
    if extractor.strict.fullmatch(text.strip()):
        return extractor.ids(text)[0]
    return None


@dataclass
class Result:
    requests: pd.DataFrame
    ariba_not_in_usf: pd.DataFrame
    ariba_without_id: pd.DataFrame
    usf_status_values: pd.DataFrame
    ariba_status_values: pd.DataFrame
    ariba_row_count: int


def reconcile(usf: pd.DataFrame, ariba: pd.DataFrame, cfg: Config) -> Result:
    ex = Extractor(cfg.id_prefix)
    usf_id_col = resolve_column(usf, cfg.usf_id_column)
    usf_status_col = resolve_column(usf, cfg.usf_status_column) if cfg.usf_status_column else None
    a_name = resolve_column(ariba, cfg.ariba_name_column)
    a_id = resolve_column(ariba, cfg.ariba_id_column)
    a_status = resolve_column(ariba, cfg.ariba_status_column) if cfg.ariba_status_column else None

    ariba = ariba.copy()
    ariba["_ids"] = ariba[a_name].map(ex.ids)
    ariba["_inactive"] = ariba[a_status].map(lambda s: norm(s) in cfg.ariba_inactive_statuses) if a_status else False
    index, malformed_index = defaultdict(list), defaultdict(list)
    for i, row in ariba.iterrows():
        for n in row["_ids"]:
            index[n].append(i)
        if not row["_ids"]:
            for n in ex.malformed_ids(row[a_name]):
                malformed_index[n].append(i)

    usf = usf.copy()
    usf["_id"] = usf[usf_id_col].map(lambda v: parse_usf_id(v, ex))
    id_counts = usf["_id"].value_counts()

    def bucket(status) -> str:
        if not cfg.eligibility_configured:
            return BUCKET_UNCONFIGURED
        s = norm(status)
        if s in cfg.eligible_statuses:
            return ELIGIBLE
        if s in cfg.pending_statuses:
            return BUCKET_PENDING
        if s in cfg.not_required_statuses:
            return BUCKET_NOT_REQUIRED
        return BUCKET_UNKNOWN

    rows = []
    for _, r in usf.iterrows():
        n = r["_id"]
        if pd.isna(n):
            n = None
        status = r[usf_status_col] if usf_status_col else ""
        b = bucket(status)
        matches = index.get(n, []) if n is not None else []
        active = [m for m in matches if not ariba.at[m, "_inactive"]]
        category, reasons = _classify(n, b, status, matches, active, malformed_index, ariba, a_name, a_status)

        if n is not None and id_counts.get(n, 0) > 1:
            reasons.append(f"ID appears {id_counts[n]} times in the USF report")
            if category == CREATED:
                category = REVIEW
        if any(len(ariba.at[m, "_ids"]) > 1 for m in matches):
            reasons.append("Ariba project name references more than one USF ID")
            if category == CREATED:
                category = REVIEW

        shown = active or matches
        rows.append({
            "Category": category,
            "Reason": "; ".join(reasons),
            "USF ID": r[usf_id_col],
            "USF Key": ex.key(int(n)) if n is not None else "",
            "USF Status": status,
            "Eligibility": b,
            "Ariba Project ID": ", ".join(str(ariba.at[m, a_id]) for m in shown),
            "Ariba Project Name": " | ".join(str(ariba.at[m, a_name]) for m in shown),
            "Ariba Status": ", ".join(str(ariba.at[m, a_status]) for m in shown) if a_status else "",
            "Ariba Match Count": len(matches),
            "_id": n,
            "_usf_row": r.drop(labels=["_id"]),
        })

    requests = pd.DataFrame(rows)
    if not requests.empty:
        display = _display_columns(usf.drop(columns=["_id"]), cfg, exclude={usf_id_col, usf_status_col})
        extra = pd.DataFrame([row[display] for row in requests.pop("_usf_row")], columns=display).reset_index(drop=True)
        requests = pd.concat([requests.reset_index(drop=True), extra.add_prefix("USF: ")], axis=1)
        requests["_order"] = requests["Category"].map(CATEGORY_ORDER.index)
        requests = requests.sort_values(["_order", "_id"], na_position="first").drop(columns="_order").reset_index(drop=True)

    usf_ids = set(usf["_id"].dropna().astype(int))
    base_cols = [a_id, a_name] + ([a_status] if a_status else [])
    not_in_usf, without_id = [], []
    for _, row in ariba.iterrows():
        if row["_ids"]:
            unknown = [ex.key(n) for n in row["_ids"] if n not in usf_ids]
            if unknown:
                not_in_usf.append({**row[base_cols].to_dict(), "USF Key(s) Not in USF Report": ", ".join(unknown)})
        else:
            bad = ex.malformed_ids(row[a_name])
            reason = (f"Malformed USF ID (expected {cfg.id_prefix}<number>), possible: "
                      + ", ".join(ex.key(n) for n in bad)) if bad else "No USF ID in project name"
            without_id.append({**row[base_cols].to_dict(), "Reason": reason})

    return Result(
        requests=requests,
        ariba_not_in_usf=pd.DataFrame(not_in_usf, columns=base_cols + ["USF Key(s) Not in USF Report"]),
        ariba_without_id=pd.DataFrame(without_id, columns=base_cols + ["Reason"]),
        usf_status_values=_usf_status_table(usf, usf_status_col, cfg),
        ariba_status_values=_ariba_status_table(ariba, a_status, cfg),
        ariba_row_count=len(ariba),
    )


def _classify(n, b, status, matches, active, malformed_index, ariba, a_name, a_status):
    if n is None:
        return REVIEW, ["USF ID is blank or not a number"]
    if len(active) > 1:
        return DUPLICATE, [f"{len(active)} active Ariba SRs reference this ID"]
    if len(active) == 1:
        if b in (ELIGIBLE, BUCKET_UNCONFIGURED):
            return CREATED, []
        return REVIEW, [f"Ariba SR exists but USF status is '{status}'"]
    if matches:
        if b == BUCKET_NOT_REQUIRED:
            return NOT_REQUIRED, ["Only inactive Ariba SR found"]
        statuses = ", ".join(sorted({str(ariba.at[m, a_status]) for m in matches})) if a_status else ""
        return REVIEW, [f"Only inactive Ariba SR found ({statuses})"]
    if n in malformed_index and b != BUCKET_NOT_REQUIRED:
        names = " | ".join(str(ariba.at[m, a_name]) for m in malformed_index[n])
        return REVIEW, [f"Possible match with malformed ID in Ariba project name: {names}"]
    return {
        ELIGIBLE: (MISSING, []),
        BUCKET_PENDING: (PENDING, []),
        BUCKET_NOT_REQUIRED: (NOT_REQUIRED, []),
        BUCKET_UNKNOWN: (UNKNOWN, [f"Status '{status}' is not listed in the SETTINGS block"]),
        BUCKET_UNCONFIGURED: (UNCONFIRMED, ["Set ELIGIBLE_STATUSES to confirm"]),
    }[b]


def _display_columns(usf, cfg, exclude):
    if cfg.display_columns:
        return [resolve_column(usf, c) for c in cfg.display_columns]
    return [c for c in usf.columns if c not in exclude]


def _usf_status_table(usf, col, cfg):
    if not col:
        return pd.DataFrame(columns=["USF Status", "Count", "Configured As"])
    lists = [("Eligible", cfg.eligible_statuses), ("Pending", cfg.pending_statuses),
             ("Not required", cfg.not_required_statuses)]
    counts = usf[col].fillna("").map(lambda s: " ".join(str(s).split())).value_counts()
    return pd.DataFrame([
        {"USF Status": s or "(blank)", "Count": c,
         "Configured As": next((name for name, vals in lists if norm(s) in vals), "NOT CONFIGURED")}
        for s, c in counts.items()])


def _ariba_status_table(ariba, col, cfg):
    if not col:
        return pd.DataFrame(columns=["Ariba Status", "Count", "Treated As"])
    counts = ariba[col].fillna("").map(lambda s: " ".join(str(s).split())).value_counts()
    return pd.DataFrame([
        {"Ariba Status": s or "(blank)", "Count": c,
         "Treated As": "Inactive (not a valid SR)" if norm(s) in cfg.ariba_inactive_statuses else "Active"}
        for s, c in counts.items()])


# =============================================================================
# Run-to-run tracking (the Ariba report has no creation date)
# =============================================================================

HISTORY_COLUMNS = ["USF Key", "First Seen", "First Detected Missing", "First Observed in Ariba",
                   "Last Category", "Last Run"]


def apply_history(requests: pd.DataFrame, path: Path, today: date) -> pd.DataFrame:
    hist = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=HISTORY_COLUMNS)
    hist = hist.reindex(columns=HISTORY_COLUMNS, fill_value="").set_index("USF Key")
    t = today.isoformat()
    first_missing, first_observed, outstanding = [], [], []
    for _, r in requests.iterrows():
        key = r["USF Key"]
        if not key:
            first_missing.append("")
            first_observed.append("")
            outstanding.append(None)
            continue
        if key not in hist.index:
            hist.loc[key] = {c: "" for c in HISTORY_COLUMNS[1:]}
            hist.at[key, "First Seen"] = t
        if r["Category"] == MISSING and not hist.at[key, "First Detected Missing"]:
            hist.at[key, "First Detected Missing"] = t
        if r["Ariba Match Count"] > 0 and not hist.at[key, "First Observed in Ariba"]:
            hist.at[key, "First Observed in Ariba"] = t
        hist.at[key, "Last Category"] = r["Category"]
        hist.at[key, "Last Run"] = t
        fm, fo = hist.at[key, "First Detected Missing"], hist.at[key, "First Observed in Ariba"]
        first_missing.append(fm)
        first_observed.append(fo)
        outstanding.append(int(np.busday_count(fm, t)) if r["Category"] == MISSING and fm else None)

    requests = requests.copy()
    pos = requests.columns.get_loc("Ariba Match Count") + 1
    requests.insert(pos, "First Detected Missing", first_missing)
    requests.insert(pos + 1, "First Observed in Ariba", first_observed)
    requests.insert(pos + 2, "Business Days Missing", outstanding)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist.reset_index().to_csv(path, index=False)
    return requests


# =============================================================================
# Output workbook
# =============================================================================

HEADER_FILL = PatternFill("solid", fgColor="1A5336")
HEADER_FONT = Font(bold=True, color="FFFFFF")
CATEGORY_FILLS = {
    MISSING: "F8CBAD", DUPLICATE: "F8CBAD", REVIEW: "FFE699", UNKNOWN: "FFE699", UNCONFIRMED: "FFE699",
    CREATED: "C6EFCE", PENDING: "DDEBF7", NOT_REQUIRED: "EDEDED",
}


def summary_text(result: Result, run_at: datetime) -> str:
    req = result.requests
    counts = req["Category"].value_counts() if not req.empty else pd.Series(dtype=int)
    missing = req.loc[req["Category"] == MISSING, "USF Key"].tolist() if not req.empty else []
    lines = [f"Daily SR Reconciliation: {run_at:%B %d, %Y at %I:%M %p}", f"USF requests checked: {len(req)}"]
    lines += [f"{c}: {int(counts.get(c, 0))}" for c in CATEGORY_ORDER if counts.get(c, 0)]
    lines.append(f"Ariba SRs not in USF report: {len(result.ariba_not_in_usf)}")
    lines.append(f"Ariba SRs without a valid USF ID: {len(result.ariba_without_id)}")
    if missing:
        lines.append("Missing requests: " + ", ".join(missing))
    return "\n".join(lines)


def write_workbook(result: Result, path: Path, run_info: list, warnings: list, summary: str) -> None:
    req = result.requests.drop(columns=["_id"], errors="ignore")
    action = req[req["Category"].isin(ACTION_CATEGORIES)] if not req.empty else req
    counts = req["Category"].value_counts() if not req.empty else pd.Series(dtype=int)

    rows = [("RUN DETAILS", "")] + run_info + [("", ""), ("RESULTS", "")]
    rows += [(c, int(counts.get(c, 0))) for c in CATEGORY_ORDER]
    rows += [("Ariba SRs not in USF report", len(result.ariba_not_in_usf)),
             ("Ariba SRs without a valid USF ID", len(result.ariba_without_id)), ("", "")]
    if warnings:
        rows += [("WARNINGS", "")] + [("Warning", w) for w in warnings] + [("", "")]
    rows += [("SUMMARY TO SHARE", "")] + [("", line) for line in summary.splitlines()]

    sheets = {
        "Summary": pd.DataFrame(rows, columns=["Item", "Value"]),
        "Action Required": action,
        "All Requests": req,
        "Ariba Not in USF": result.ariba_not_in_usf,
        "Ariba Without USF ID": result.ariba_without_id,
        "USF Status Values": result.usf_status_values,
        "Ariba Status Values": result.ariba_status_values,
    }
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name, index=False)
                _style(xw.sheets[name], df, name == "Summary")
    except PermissionError as exc:
        raise InputError(f"Cannot save {path.name}. Close it in Excel and run again.") from exc


def _style(ws, df, is_summary):
    for cell in ws[1]:
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
    ws.freeze_panes = "A2"
    if ws.max_row > 1 and not is_summary:
        ws.auto_filter.ref = ws.dimensions
    for i, col in enumerate(df.columns, start=1):
        sample = [str(col)] + [str(v) for v in df[col].head(300).tolist()]
        ws.column_dimensions[get_column_letter(i)].width = min(max(len(s) for s in sample) + 2, 60)
    if is_summary:
        for item, value in ws.iter_rows(min_row=2, max_col=2):
            if item.value and str(item.value).isupper():
                item.font = Font(bold=True, color="1A5336")
            if item.value in CATEGORY_FILLS:
                item.fill = PatternFill("solid", fgColor=CATEGORY_FILLS[item.value])
            if item.value == "Warning":
                item.font = value.font = Font(bold=True, color="C00000")
            value.alignment = Alignment(horizontal="left", wrap_text=True)
        ws.column_dimensions["B"].width = 100
        return
    if "Category" in df.columns:
        idx = list(df.columns).index("Category") + 1
        for (cell,) in ws.iter_rows(min_row=2, min_col=idx, max_col=idx):
            if cell.value in CATEGORY_FILLS:
                cell.fill = PatternFill("solid", fgColor=CATEGORY_FILLS[cell.value])


# =============================================================================
# Main
# =============================================================================

def pick_files():
    """File pickers for double-click use. Returns (usf, ariba) or (None, None) if cancelled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise InputError("No file picker available. Run with --usf FILE --ariba FILE.")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    types = [("Reports", "*.xlsx *.xlsm *.csv"), ("All files", "*.*")]
    usf = filedialog.askopenfilename(title="Choose the USF export", filetypes=types)
    ariba = filedialog.askopenfilename(title="Choose the Ariba SR report", filetypes=types) if usf else ""
    root.destroy()
    return (Path(usf), Path(ariba)) if usf and ariba else (None, None)


def main(argv=None, cfg: "Config | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Reconcile the USF report against the Ariba SR report.")
    ap.add_argument("--usf", type=Path, help="USF export file")
    ap.add_argument("--ariba", type=Path, help="Ariba SR report file")
    ap.add_argument("--out", type=Path, help="Folder for the results (default: same folder as the USF file)")
    ap.add_argument("--no-open", action="store_true", help="Do not open the workbook when finished")
    args = ap.parse_args(argv)

    try:
        cfg = (cfg or Config()).validate()
        usf_path, ariba_path = args.usf, args.ariba
        if not (usf_path and ariba_path):
            usf_path, ariba_path = pick_files()
            if not usf_path:
                print("Cancelled.")
                return 1
        for p in (usf_path, ariba_path):
            if not p.exists():
                raise InputError(f"File not found: {p}")
        print(f"USF report:   {usf_path.name}")
        print(f"Ariba report: {ariba_path.name}")

        usf = read_report(usf_path, [cfg.usf_id_column, cfg.usf_status_column], "USF")
        ariba = read_report(ariba_path, [cfg.ariba_name_column, cfg.ariba_id_column, cfg.ariba_status_column], "Ariba")
        result = reconcile(usf, ariba, cfg)
        run_at = datetime.now()
        result.requests = apply_history(result.requests, cfg.history_file, date.today())

        warnings = list(cfg.warnings)
        for label, p in (("USF", usf_path), ("Ariba", ariba_path)):
            modified = datetime.fromtimestamp(p.stat().st_mtime)
            if modified.date() != run_at.date():
                warnings.append(f"The {label} file was last saved {modified:%Y-%m-%d %H:%M}, not today. "
                                f"Results may be out of date.")
        sv = result.usf_status_values
        if cfg.eligibility_configured and len(sv):
            new = sv.loc[sv["Configured As"] == "NOT CONFIGURED", "USF Status"].tolist()
            if new:
                warnings.append(f"USF statuses not listed in the SETTINGS block: {new}")

        summary = summary_text(result, run_at)
        run_info = [
            ("Run at", f"{run_at:%Y-%m-%d %H:%M}"),
            ("USF file", str(usf_path)),
            ("USF file last saved", f"{datetime.fromtimestamp(usf_path.stat().st_mtime):%Y-%m-%d %H:%M}"),
            ("USF rows", str(len(usf))),
            ("Ariba file", str(ariba_path)),
            ("Ariba file last saved", f"{datetime.fromtimestamp(ariba_path.stat().st_mtime):%Y-%m-%d %H:%M}"),
            ("Ariba rows", str(result.ariba_row_count)),
            ("Eligible USF statuses", ", ".join(sorted(cfg.eligible_statuses)) or "(not configured)"),
        ]
        out_dir = args.out or usf_path.resolve().parent
        out = out_dir / f"USF_Ariba_Reconciliation_{run_at:%Y-%m-%d_%H%M}.xlsx"
        write_workbook(result, out, run_info, warnings, summary)
    except (ConfigError, InputError) as exc:
        print(f"\nPROBLEM: {exc}\n")
        return 1

    print("\n" + summary)
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"\nResults saved: {out}")
    if not args.no_open:
        try:
            if sys.platform == "win32":
                os.startfile(out)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(out)], check=False)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    code = main()
    if sys.platform == "win32" and len(sys.argv) == 1:
        input("\nPress Enter to close.")
    sys.exit(code)
