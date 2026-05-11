```
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Applied AI/ML Hub - Banner</title>
  <style>
    /* Safe, self-contained styles. Works inside Confluence HTML macro. */
    .ai-banner-link{ text-decoration: none; display: block; width:100%; }
    .ai-banner{
      box-sizing: border-box;
      width: 100%;
      margin: 0;
      padding: 18px 24px;
      border-radius: 0;
      background: linear-gradient(180deg,#012048 0%, #002d6a 100%);
      color: #ffffff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.18);
      border-top: 1px solid rgba(255,255,255,0.04);
      border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .ai-banner__left{
      display:flex;
      align-items: center;
      gap: 16px;
    }
    .ai-banner__logo{
      flex: 0 0 auto;
      width: 56px;
      height: 56px;
      border-radius: 8px;
      background: rgba(255,255,255,0.06);
      display:flex;
      align-items:center;
      justify-content:center;
      font-weight:700;
      font-size:18px;
      letter-spacing:0.6px;
    }
    .ai-banner__title{
      font-size:20px;
      line-height:1.05;
      margin:0;
      padding:0;
      font-weight:700;
    }
    .ai-banner__subtitle{
      margin:0;
      font-size:12px;
      opacity:0.9;
      font-weight:500;
    }
    .ai-banner__cta{
      flex:0 0 auto;
      background: rgba(255,255,255,0.06);
      padding:10px 16px;
      border-radius:8px;
      font-weight:600;
      font-size:13px;
      border: 1px solid rgba(255,255,255,0.06);
      transition: transform 120ms ease, box-shadow 120ms ease;
    }
    .ai-banner__cta:active{ transform: translateY(1px); }
    .ai-banner-link:focus .ai-banner, .ai-banner-link:hover .ai-banner{ box-shadow: 0 10px 28px rgba(0,0,0,0.25); transform: translateY(-2px); }

    @media (max-width:720px){
      .ai-banner{ flex-direction: column; align-items:flex-start; padding:14px; }
      .ai-banner__left{ gap:12px; }
      .ai-banner__title{ font-size:18px; }
      .ai-banner__logo{ width:48px; height:48px; font-size:16px; }
      .ai-banner__cta{ width:100%; text-align:center; }
    }
  </style>
</head>
<body>
  <!-- Replace the href value with the URL you want the banner to open -->
  <a class="ai-banner-link" href="YOUR_URL_HERE" target="_blank" rel="noopener noreferrer">
    <div class="ai-banner" role="banner" aria-label="Applied AI/ML Hub - AI Sandbox">
      <div class="ai-banner__left">
        <div class="ai-banner__logo" aria-hidden="true">AI</div>
        <div>
          <h1 class="ai-banner__title">Applied AI/ML Hub</h1>
          <p class="ai-banner__subtitle">AI Sandbox</p>
        </div>
      </div>
      <div class="ai-banner__cta" role="button" aria-hidden="true">Open Sandbox</div>
    </div>
  </a>
</body>
</html>
```
---
```
# ============================================================
# Check-ADUserStatus.ps1
#
# Reads a CSV/Excel file with columns: ID, Email
# Queries Active Directory for each user's enabled/disabled status
# Outputs the same file with an added "Account Status" column
# ============================================================

param (
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [Parameter(Mandatory = $false)]
    [string]$OutputFile
)

# --- Resolve output path ---
if (-not $OutputFile) {
    $baseName  = [System.IO.Path]::GetFileNameWithoutExtension($InputFile)
    $extension = [System.IO.Path]::GetExtension($InputFile)
    $directory = [System.IO.Path]::GetDirectoryName($InputFile)
    $OutputFile = Join-Path $directory ($baseName + "_ADStatus" + $extension)
}

# --- Load input file (supports .csv, .xlsx via Export-Excel or plain CSV) ---
$extension = [System.IO.Path]::GetExtension($InputFile).ToLower()

if ($extension -eq ".csv") {
    $users = Import-Csv -Path $InputFile
} elseif ($extension -in @(".xlsx", ".xls")) {
    # Requires the ImportExcel module: Install-Module -Name ImportExcel
    if (-not (Get-Module -ListAvailable -Name ImportExcel)) {
        Write-Error "The 'ImportExcel' module is required for .xlsx files. Install it with: Install-Module -Name ImportExcel"
        exit 1
    }
    Import-Module ImportExcel
    $users = Import-Excel -Path $InputFile
} else {
    Write-Error "Unsupported file type '$extension'. Please provide a .csv or .xlsx file."
    exit 1
}

# --- Detect column names flexibly (case-insensitive) ---
$allColumns = $users[0].PSObject.Properties.Name

$idColumn = $allColumns | Where-Object { $_ -match '^id$|^user.?id$|^employee.?id$|^username$|^sam' } | Select-Object -First 1
$emailColumn = $allColumns | Where-Object { $_ -match '^email$|^mail$|^e.?mail' } | Select-Object -First 1

if (-not $idColumn) {
    Write-Warning "Could not auto-detect the ID column. Falling back to first column: '$($allColumns[0])'"
    $idColumn = $allColumns[0]
}
if (-not $emailColumn) {
    Write-Warning "Could not auto-detect the Email column. Falling back to second column: '$($allColumns[1])'"
    $emailColumn = $allColumns[1]
}

Write-Host "Using ID column     : $idColumn"
Write-Host "Using Email column  : $emailColumn"
Write-Host "Processing $($users.Count) users..."
Write-Host ""

# --- Process each user ---
$results = foreach ($user in $users) {
    $userId = $user.$idColumn
    $email  = $user.$emailColumn

    $status = "Unknown"

    if ([string]::IsNullOrWhiteSpace($userId)) {
        $status = "No ID Provided"
    } else {
        try {
            # Same approach as your working command:
            # Get-ADUser personID -Properties PrimaryGroup
            $adUser = Get-ADUser $userId -Properties Enabled -ErrorAction Stop

            if ($adUser.Enabled -eq $true) {
                $status = "Active"
            } else {
                $status = "Disabled"
            }
        } catch [Microsoft.ActiveDirectory.Management.ADIdentityNotFoundException] {
            $status = "Not Found in AD"
        } catch {
            $status = "Error: $($_.Exception.Message)"
        }
    }

    Write-Host "  $userId | $email | $status"

    # Build output row preserving all original columns
    $row = [ordered]@{}
    foreach ($col in $allColumns) {
        $row[$col] = $user.$col
    }
    $row["Account Status"] = $status

    [PSCustomObject]$row
}

Write-Host ""
Write-Host "Writing output to: $OutputFile"

# --- Export results ---
if ($extension -eq ".csv") {
    $results | Export-Csv -Path $OutputFile -NoTypeInformation -Encoding UTF8
} else {
    $results | Export-Excel -Path $OutputFile -AutoSize -AutoFilter -WorksheetName "AD Status" `
        -TableName "UserStatus" -TableStyle Medium2 `
        -ConditionalText $(
            New-ConditionalText -Text "Active"          -BackgroundColor "#C6EFCE" -ConditionalTextColor "#276221"
            New-ConditionalText -Text "Disabled"        -BackgroundColor "#FFCCCC" -ConditionalTextColor "#9C0006"
            New-ConditionalText -Text "Not Found in AD" -BackgroundColor "#FFEB9C" -ConditionalTextColor "#9C5700"
        )
}

Write-Host "Done."

# --- Summary ---
$summary = $results | Group-Object "Account Status" | Select-Object Name, Count
Write-Host ""
Write-Host "=== Summary ==="
$summary | Format-Table -AutoSize
```
---
# For a CSV input:
.\Check-ADUserStatus.ps1 -InputFile "C:\users\people.csv"

# For an Excel input:
.\Check-ADUserStatus.ps1 -InputFile "C:\users\people.xlsx"

# With a custom output path:
.\Check-ADUserStatus.ps1 -InputFile "people.csv" -OutputFile "C:\output\results.csv"
---
Get-ADUser personID -Properties Enabled | Select-Object SamAccountName, Enabled
----------------------

```
<!DOCTYPE html>
<html>
<body style="margin:0; padding:0; background-color:#f4f5f7; font-family: Arial, sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f5f7; padding: 40px 0;">
    <tr>
      <td align="center">

        <!-- Email Card -->
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff; border-radius:8px; overflow:hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">

          <!-- Header Banner -->
          <tr>
            <td style="background-color:#0052CC; padding: 32px 40px; text-align:center;">
              <h1 style="color:#ffffff; margin:0; font-size:22px; font-weight:700;">
                🛠️ New Support Request
              </h1>
              <p style="color:#B3D4FF; margin:8px 0 0 0; font-size:14px;">
                A new issue has been submitted and requires attention
              </p>
            </td>
          </tr>

          <!-- Submitted By Banner -->
          <tr>
            <td style="background-color:#DEEBFF; padding: 12px 40px;">
              <p style="margin:0; font-size:13px; color:#0052CC;">
                <strong>Submitted by:</strong> [entry._creator] &nbsp;|&nbsp;
                <strong>Date:</strong> [entry._created]
              </p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding: 32px 40px;">

              <!-- Issue Title -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px;">
                <tr>
                  <td style="border-left: 4px solid #0052CC; padding-left:12px;">
                    <p style="margin:0 0 4px 0; font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#6B778C;">Issue Title</p>
                    <p style="margin:0; font-size:18px; font-weight:700; color:#172B4D;">[entry.title]</p>
                  </td>
                </tr>
              </table>

              <!-- Priority Badge -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px;">
                <tr>
                  <td>
                    <p style="margin:0 0 8px 0; font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#6B778C;">Priority</p>
                    <span style="display:inline-block; background-color:#FF5630; color:#ffffff; padding:4px 16px; border-radius:20px; font-size:13px; font-weight:700;">[entry.priority]</span>
                  </td>
                </tr>
              </table>

              <!-- Divider -->
              <hr style="border:none; border-top:1px solid #EBECF0; margin: 0 0 24px 0;" />

              <!-- Description -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px;">
                <tr>
                  <td style="background-color:#F4F5F7; border-radius:6px; padding:16px;">
                    <p style="margin:0 0 8px 0; font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#6B778C;">Issue Description</p>
                    <p style="margin:0; font-size:14px; color:#172B4D; line-height:1.6;">[entry.message]</p>
                  </td>
                </tr>
              </table>

              <!-- Department and any other fields -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px; border:1px solid #EBECF0; border-radius:6px; overflow:hidden;">
                <tr style="background-color:#F4F5F7;">
                  <td width="40%" style="padding:10px 16px; font-size:12px; font-weight:700; color:#6B778C; text-transform:uppercase;">Field</td>
                  <td width="60%" style="padding:10px 16px; font-size:12px; font-weight:700; color:#6B778C; text-transform:uppercase;">Value</td>
                </tr>
                <tr style="border-top:1px solid #EBECF0;">
                  <td style="padding:12px 16px; font-size:13px; color:#6B778C;">Department</td>
                  <td style="padding:12px 16px; font-size:13px; color:#172B4D; font-weight:600;">[entry.department]</td>
                </tr>
                <tr style="border-top:1px solid #EBECF0; background-color:#FAFBFC;">
                  <td style="padding:12px 16px; font-size:13px; color:#6B778C;">Steps to Reproduce</td>
                  <td style="padding:12px 16px; font-size:13px; color:#172B4D;">[entry.steps]</td>
                </tr>
                <tr style="border-top:1px solid #EBECF0;">
                  <td style="padding:12px 16px; font-size:13px; color:#6B778C;">Expected Behaviour</td>
                  <td style="padding:12px 16px; font-size:13px; color:#172B4D;">[entry.expected]</td>
                </tr>
                <tr style="border-top:1px solid #EBECF0; background-color:#FAFBFC;">
                  <td style="padding:12px 16px; font-size:13px; color:#6B778C;">Actual Behaviour</td>
                  <td style="padding:12px 16px; font-size:13px; color:#172B4D;">[entry.actual]</td>
                </tr>
              </table>

            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color:#F4F5F7; padding:20px 40px; text-align:center; border-top:1px solid #EBECF0;">
              <p style="margin:0; font-size:12px; color:#6B778C;">
                This is an automated notification from your IT Support Form.<br/>
                Please do not reply to this email directly.
              </p>
            </td>
          </tr>

        </table>
        <!-- End Email Card -->

      </td>
    </tr>
  </table>

</body>
</html>
```
---
```
"""
lookup_user_domains.py
----------------------
Reads an Excel file with a "TD User ID" column, queries Active Directory (LDAP)
across multiple domains in parallel to find each user's domain, then writes
the result to a new Excel file with a "Domain" column appended.

Usage:
    python lookup_user_domains.py

Requirements:
    pip install pandas openpyxl ldap3 tqdm
"""

import pandas as pd
from ldap3 import Server, Connection, AUTO_BIND_NO_TLS, NTLM, ALL_ATTRIBUTES, SUBTREE
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import os
import json
import getpass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE   = r"C:\path\to\your\input_file.xlsx"   # <-- change this
OUTPUT_FILE  = r"C:\path\to\your\output_file.xlsx"  # <-- change this
CHECKPOINT   = r"C:\path\to\checkpoint.json"         # progress saved here

# Your AD credentials (you will be prompted at runtime if left blank)
AD_USERNAME  = ""   # e.g. "TDBFG\\your.name"  or leave blank to prompt
AD_PASSWORD  = ""   # leave blank to prompt securely

# Domain controllers — add/remove as needed.
# Format: { "DomainLabel": "domain_controller_hostname_or_ip" }
DOMAINS = {
    "TDBFG":         "tdbfg.ad.td.com",        # <-- replace with real DC hostnames
    "TDSecurities":  "tdsecurities.ad.td.com",
    "TDCT":          "tdct.ad.td.com",
    "AMTD":          "amtd.ad.td.com",
}

# Base DNs to search in each domain (adjust to match your AD structure)
SEARCH_BASES = {
    "TDBFG":         "DC=tdbfg,DC=ad,DC=td,DC=com",
    "TDSecurities":  "DC=tdsecurities,DC=ad,DC=td,DC=com",
    "TDCT":          "DC=tdct,DC=ad,DC=td,DC=com",
    "AMTD":          "DC=amtd,DC=ad,DC=td,DC=com",
}

LDAP_PORT        = 389      # 389 = LDAP, 636 = LDAPS
USE_SSL          = False    # set True if using port 636
MAX_WORKERS      = 8        # parallel threads (tune to your network; 8–16 is usually safe)
BATCH_SAVE_EVERY = 500      # save checkpoint every N users

# ─────────────────────────────────────────────────────────────────────────────
# LDAP helpers
# ─────────────────────────────────────────────────────────────────────────────

# Thread-local storage: one Connection per thread per domain
_thread_local = threading.local()


def _get_connection(domain_label: str, username: str, password: str) -> Connection | None:
    """Return a cached (thread-local) LDAP connection for the given domain."""
    if not hasattr(_thread_local, "conns"):
        _thread_local.conns = {}

    if domain_label not in _thread_local.conns:
        dc = DOMAINS[domain_label]
        try:
            server = Server(dc, port=LDAP_PORT, use_ssl=USE_SSL, connect_timeout=5)
            conn = Connection(
                server,
                user=username,
                password=password,
                authentication=NTLM,
                auto_bind=AUTO_BIND_NO_TLS,
                receive_timeout=10,
            )
            _thread_local.conns[domain_label] = conn
        except Exception:
            _thread_local.conns[domain_label] = None   # mark as unavailable

    return _thread_local.conns[domain_label]


def find_user_domain(sam_account: str, username: str, password: str) -> str:
    """
    Search each domain for sam_account.
    Returns the domain label if found and account is enabled, else "NOT FOUND".
    Skips to the next domain immediately on any error.
    """
    sam_account = sam_account.strip()
    if not sam_account:
        return "BLANK"

    ldap_filter = f"(&(objectClass=user)(sAMAccountName={sam_account}))"

    for domain_label, base_dn in SEARCH_BASES.items():
        conn = _get_connection(domain_label, username, password)
        if conn is None:
            continue
        try:
            conn.search(
                search_base=base_dn,
                search_filter=ldap_filter,
                search_scope=SUBTREE,
                attributes=["sAMAccountName", "userAccountControl"],
                time_limit=8,
                size_limit=1,
            )
            if conn.entries:
                # Check if account is disabled (userAccountControl bit 2 = disabled)
                uac = conn.entries[0].userAccountControl.value
                if isinstance(uac, int) and (uac & 0x2):
                    return f"{domain_label} (DISABLED)"
                return domain_label
        except Exception:
            # Connection dropped — clear it so it reconnects next time
            _thread_local.conns.pop(domain_label, None)
            continue

    return "NOT FOUND"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(results: dict):
    with open(CHECKPOINT, "w") as f:
        json.dump(results, f)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Credentials ──────────────────────────────────────────────────────────
    username = AD_USERNAME or input("AD Username (e.g. TDBFG\\your.name): ").strip()
    password = AD_PASSWORD or getpass.getpass("AD Password: ")

    # ── Load Excel ───────────────────────────────────────────────────────────
    print(f"\nReading {INPUT_FILE} ...")
    df = pd.read_excel(INPUT_FILE, dtype=str)

    col = "TD User ID"
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Columns found: {list(df.columns)}")

    user_ids = df[col].fillna("").tolist()
    total    = len(user_ids)
    print(f"Loaded {total:,} rows.\n")

    # ── Resume from checkpoint ───────────────────────────────────────────────
    results = load_checkpoint()
    remaining = [(i, uid) for i, uid in enumerate(user_ids) if str(i) not in results]
    print(f"{len(results):,} already done from checkpoint. {len(remaining):,} to process.\n")

    # ── Parallel lookup ───────────────────────────────────────────────────────
    lock = threading.Lock()
    done_since_save = 0

    def lookup(args):
        idx, uid = args
        domain = find_user_domain(uid, username, password)
        return str(idx), domain

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(lookup, item): item for item in remaining}

        with tqdm(total=len(remaining), unit="user", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                idx_str, domain = future.result()
                with lock:
                    results[idx_str] = domain
                    done_since_save += 1
                    bar.set_postfix_str(f"last={domain}", refresh=False)
                    bar.update(1)

                    if done_since_save >= BATCH_SAVE_EVERY:
                        save_checkpoint(results)
                        done_since_save = 0

    save_checkpoint(results)   # final save

    # ── Write output Excel ────────────────────────────────────────────────────
    print(f"\nWriting results to {OUTPUT_FILE} ...")
    df["Domain"] = [results.get(str(i), "NOT FOUND") for i in range(total)]

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Users")
        ws = writer.sheets["Users"]

        # Auto-size columns
        for col_cells in ws.columns:
            max_len = max((len(str(c.value or "")) for c in col_cells), default=10)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 60)

        # Highlight "NOT FOUND" rows in light red
        from openpyxl.styles import PatternFill
        red_fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
        domain_col_idx = df.columns.get_loc("Domain") + 1   # 1-based
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            if row[domain_col_idx - 1].value in ("NOT FOUND", "BLANK"):
                for cell in row:
                    cell.fill = red_fill

    print(f"\nDone! Output saved to: {OUTPUT_FILE}")
    print(f"Checkpoint file (safe to delete): {CHECKPOINT}\n")

    # ── Summary stats ─────────────────────────────────────────────────────────
    domain_col = df["Domain"]
    print("─── Domain Breakdown ───────────────────────────────")
    print(domain_col.value_counts().to_string())
    print("────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()

```