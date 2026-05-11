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
Reads an Excel file with a "TD User ID" column, uses PowerShell Get-ADUser
(running as your current Windows session) to find which domain each user
belongs to. No LDAP, no credentials, no DC connections.

Usage:
    python lookup_user_domains.py

Requirements:
    pip install pandas openpyxl tqdm
"""

import pandas as pd
import subprocess
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — only edit these
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE       = r"C:\path\to\your\input_file.xlsx"   # <-- change this
OUTPUT_FILE      = r"C:\path\to\your\output_file.xlsx"  # <-- change this
CHECKPOINT       = r"C:\path\to\checkpoint.json"         # <-- change this

MAX_WORKERS      = 20    # how many parallel PowerShell queries to run at once
BATCH_SIZE       = 50    # users per PowerShell call (batching = much faster)
BATCH_SAVE_EVERY = 10    # save checkpoint every N batches

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Discover all domains in the forest via PowerShell
# ─────────────────────────────────────────────────────────────────────────────

def get_all_domains() -> list[str]:
    """Returns list of domain DNS names, e.g. ['tdbfg.com', 'tdsecurities.com', ...]"""
    print("\n[1/3] Discovering all domains in the forest...")
    ps = r"""
$ErrorActionPreference = 'SilentlyContinue'
try {
    (Get-ADForest).Domains | ConvertTo-Json
} catch {
    @((Get-ADDomain).DNSRoot) | ConvertTo-Json
}
"""
    r = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True, text=True, timeout=30
    )
    raw = r.stdout.strip()
    if not raw:
        raise RuntimeError(f"Could not discover domains.\nSTDERR: {r.stderr.strip()}")

    data = json.loads(raw)
    domains = [data] if isinstance(data, str) else data
    print(f"  Found {len(domains)} domain(s): {', '.join(domains)}")
    return domains


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Batch lookup: given a list of sAMAccountNames, query each domain
#           and return a dict of { username -> domain_netbios }
# ─────────────────────────────────────────────────────────────────────────────

def lookup_batch(user_ids: list[str], domains: list[str]) -> dict[str, str]:
    """
    Runs a single PowerShell script that checks each domain for each user.
    Returns dict: { "jsmith": "TDBFG", "bjones": "TDSecurities", ... }
    """
    # Build a JSON array of usernames to pass into PS
    users_json = json.dumps(user_ids)

    # Build the list of domains as a PS array literal
    domain_list = ", ".join(f'"{d}"' for d in domains)

    ps = f"""
$ErrorActionPreference = 'SilentlyContinue'
$users   = '{users_json}' | ConvertFrom-Json
$domains = @({domain_list})
$results = @{{}}

foreach ($user in $users) {{
    $found = $false
    foreach ($domain in $domains) {{
        try {{
            $obj = Get-ADUser -Identity $user -Server $domain -ErrorAction Stop |
                   Select-Object -ExpandProperty DistinguishedName
            if ($obj) {{
                # Extract NetBIOS-style domain from DistinguishedName DC= parts
                $dcParts = ($obj -split ',') | Where-Object {{ $_ -like 'DC=*' }}
                $dns = ($dcParts | ForEach-Object {{ $_ -replace 'DC=','' }}) -join '.'
                # Get NetBIOS name
                $nb = (Get-ADDomain -Server $domain).NetBIOSName
                $results[$user] = $nb
                $found = $true
                break
            }}
        }} catch {{}}
    }}
    if (-not $found) {{
        $results[$user] = 'NOT FOUND'
    }}
}}
$results | ConvertTo-Json
"""

    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=120
        )
        raw = r.stdout.strip()
        if not raw:
            # All not found
            return {u: "NOT FOUND" for u in user_ids}
        data = json.loads(raw)
        # Normalize: PS may return null for missing keys
        return {u: (data.get(u) or "NOT FOUND") for u in user_ids}
    except Exception:
        return {u: "ERROR" for u in user_ids}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}

def save_checkpoint(results: dict):
    with open(CHECKPOINT, "w") as f:
        json.dump(results, f)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Discover domains
    domains = get_all_domains()

    # Load Excel
    print(f"\n[2/3] Loading {INPUT_FILE} ...")
    df = pd.read_excel(INPUT_FILE, dtype=str)

    col = "TD User ID"
    if col not in df.columns:
        hit = [c for c in df.columns if c.strip().lower() == col.lower()]
        col = hit[0] if hit else None
    if not col:
        raise ValueError(f"'TD User ID' column not found. Available: {list(df.columns)}")

    user_ids = df[col].fillna("").str.strip().tolist()
    total    = len(user_ids)
    print(f"  {total:,} rows loaded.")

    # Load checkpoint
    results = load_checkpoint()

    # Figure out which rows still need processing
    remaining_indices = [i for i, uid in enumerate(user_ids)
                         if uid and str(i) not in results]
    # Mark blanks immediately
    for i, uid in enumerate(user_ids):
        if not uid and str(i) not in results:
            results[str(i)] = "BLANK"

    print(f"  {len(results):,} already done, {len(remaining_indices):,} remaining.\n")

    # Split remaining into batches
    batches = []
    for start in range(0, len(remaining_indices), BATCH_SIZE):
        chunk = remaining_indices[start:start + BATCH_SIZE]
        batches.append(chunk)

    print(f"[3/3] Running {len(batches)} batches of up to {BATCH_SIZE} users "
          f"across {MAX_WORKERS} parallel workers...\n")

    lock            = threading.Lock()
    done_batches    = 0

    def process_batch(index_chunk: list[int]) -> dict:
        uid_list = [user_ids[i] for i in index_chunk]
        mapping  = lookup_batch(uid_list, domains)
        # Re-key by row index
        return {str(index_chunk[j]): mapping[uid_list[j]] for j in range(len(index_chunk))}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_batch, batch): batch for batch in batches}
        with tqdm(total=len(remaining_indices), unit="user", dynamic_ncols=True) as bar:
            for f in as_completed(futures):
                batch_result = f.result()
                with lock:
                    results.update(batch_result)
                    done_batches += 1
                    bar.update(len(batch_result))
                    if done_batches % BATCH_SAVE_EVERY == 0:
                        save_checkpoint(results)

    save_checkpoint(results)

    # Write output Excel
    print(f"\n  Writing {OUTPUT_FILE} ...")
    df["Domain"] = [results.get(str(i), "NOT FOUND") for i in range(total)]

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Users")
        ws = writer.sheets["Users"]

        from openpyxl.styles import PatternFill
        red   = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
        green = PatternFill(start_color="CCFFCC", end_color="CCFFCC", fill_type="solid")
        dcol  = df.columns.get_loc("Domain")

        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            val = row[dcol].value
            if val in ("NOT FOUND", "BLANK", "ERROR"):
                for cell in row:
                    cell.fill = red
            elif val and "DISABLED" not in str(val):
                row[dcol].fill = green

        for col_cells in ws.columns:
            w = max((len(str(c.value or "")) for c in col_cells), default=10)
            ws.column_dimensions[col_cells[0].column_letter].width = min(w + 4, 60)

    print(f"\n  Done!  {OUTPUT_FILE}")
    print(f"  Checkpoint (safe to delete): {CHECKPOINT}\n")
    print("─── Domain Breakdown ───────────────────────────")
    print(df["Domain"].value_counts().to_string())
    print("────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
```