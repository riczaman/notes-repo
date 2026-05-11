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
Reads an Excel file with a "TD User ID" column, AUTO-DISCOVERS all domains
and domain controllers from Active Directory, then queries LDAP in parallel
to find each user's domain.

Usage:
    python lookup_user_domains.py

Requirements:
    pip install pandas openpyxl ldap3 tqdm
"""

import pandas as pd
from ldap3 import Server, Connection, NTLM, AUTO_BIND_NO_TLS, SUBTREE
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import subprocess
import json
import os
import re
import getpass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — only these three paths need editing
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE       = r"C:\path\to\your\input_file.xlsx"   # <-- change this
OUTPUT_FILE      = r"C:\path\to\your\output_file.xlsx"  # <-- change this
CHECKPOINT       = r"C:\path\to\checkpoint.json"         # <-- change this

AD_USERNAME      = ""   # leave blank to prompt  e.g. "TDBFG\\your.name"
AD_PASSWORD      = ""   # leave blank to prompt securely

LDAP_PORT        = 389
USE_SSL          = False
MAX_WORKERS      = 16   # parallel threads
BATCH_SAVE_EVERY = 500

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Auto-discover all forest domains + nearest DC via PowerShell
# ─────────────────────────────────────────────────────────────────────────────

def discover_domains_via_powershell() -> dict:
    print("\n[1/3] Auto-discovering domains via PowerShell (Get-ADForest / Get-ADDomain)...")

    ps = r"""
$ErrorActionPreference = 'SilentlyContinue'
$out = @()
try {
    $forest = Get-ADForest
    foreach ($d in $forest.Domains) {
        try {
            $dom = Get-ADDomain -Identity $d
            $dc  = Get-ADDomainController -DomainName $d -Discover -NextClosestSite 2>$null
            if (-not $dc) { $dc = Get-ADDomainController -DomainName $d -Discover }
            $out += [PSCustomObject]@{
                NetBIOS = $dom.NetBIOSName
                DNS     = $dom.DNSRoot
                Base    = $dom.DistinguishedName
                DC      = ($dc.HostName | Select-Object -First 1)
            }
        } catch {}
    }
} catch {
    # No forest — just grab current domain
    try {
        $dom = Get-ADDomain
        $dc  = Get-ADDomainController -Discover
        $out += [PSCustomObject]@{
            NetBIOS = $dom.NetBIOSName
            DNS     = $dom.DNSRoot
            Base    = $dom.DistinguishedName
            DC      = ($dc.HostName | Select-Object -First 1)
        }
    } catch {}
}
$out | ConvertTo-Json -Depth 2
"""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=60
        )
        raw = r.stdout.strip()
        if not raw:
            raise ValueError(f"No output from PowerShell.\nSTDERR: {r.stderr.strip()}")

        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]

        domains = {}
        for entry in data:
            label  = entry.get("NetBIOS") or entry.get("DNS", "UNKNOWN")
            dc     = entry.get("DC", "")
            base   = entry.get("Base", "")
            if isinstance(dc, list):
                dc = dc[0]
            if dc and base:
                domains[str(label)] = {"dc": str(dc), "base_dn": str(base)}

        if not domains:
            raise ValueError("Discovered 0 usable domains.")

        for label, info in domains.items():
            print(f"    {label:20s}  DC={info['dc']}")
            print(f"    {'':20s}  Base={info['base_dn']}")
        return domains

    except Exception as e:
        print(f"  PowerShell discovery failed: {e}")
        print("  Falling back to nltest...\n")
        return discover_domains_via_nltest()


def discover_domains_via_nltest() -> dict:
    """Fallback when RSAT cmdlets aren't available."""
    domains = {}

    r = subprocess.run(["nltest", "/domain_trusts", "/all_trusts"],
                       capture_output=True, text=True, timeout=30)
    names = re.findall(r'\d+: \S+ (\S+) \(', r.stdout)

    if not names:
        # Try current domain only
        r2 = subprocess.run(["nltest", "/dsgetdc:"], capture_output=True, text=True)
        m_dc  = re.search(r'DC: \\\\(\S+)', r2.stdout)
        m_dom = re.search(r'Domain: (\S+)', r2.stdout)
        if m_dc and m_dom:
            dom   = m_dom.group(1)
            label = dom.split(".")[0].upper()
            base  = ",".join(f"DC={p}" for p in dom.split("."))
            domains[label] = {"dc": m_dc.group(1), "base_dn": base}
        return domains

    for name in names:
        r3 = subprocess.run(["nltest", f"/dclist:{name}"],
                            capture_output=True, text=True, timeout=15)
        hosts = re.findall(r'\\\\(\S+)\s+\[PDC\]', r3.stdout) or \
                re.findall(r'\\\\(\S+)', r3.stdout)
        if hosts:
            dc    = hosts[0]
            base  = ",".join(f"DC={p}" for p in name.split("."))
            label = name.split(".")[0].upper()
            domains[label] = {"dc": dc, "base_dn": base}

    print(f"  Found {len(domains)} domain(s) via nltest.")
    return domains


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Validate we can actually bind to each DC
# ─────────────────────────────────────────────────────────────────────────────

def validate_domains(domains: dict, username: str, password: str) -> dict:
    print(f"\n[2/3] Testing LDAP connectivity ({len(domains)} domain(s))...")
    valid = {}
    for label, info in domains.items():
        try:
            srv  = Server(info["dc"], port=LDAP_PORT, use_ssl=USE_SSL, connect_timeout=6)
            conn = Connection(srv, user=username, password=password,
                              authentication=NTLM, auto_bind=AUTO_BIND_NO_TLS,
                              receive_timeout=8)
            # Quick sanity search
            conn.search(info["base_dn"], "(objectClass=domain)", SUBTREE,
                        attributes=["distinguishedName"], size_limit=1, time_limit=5)
            conn.unbind()
            print(f"  OK   {label:20s}  {info['dc']}")
            valid[label] = info
        except Exception as e:
            print(f"  SKIP {label:20s}  {info['dc']}  ({e})")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Parallel LDAP user lookups
# ─────────────────────────────────────────────────────────────────────────────

_tl = threading.local()

def _conn(label: str, info: dict, username: str, password: str):
    if not hasattr(_tl, "c"):
        _tl.c = {}
    if label not in _tl.c:
        try:
            srv  = Server(info["dc"], port=LDAP_PORT, use_ssl=USE_SSL, connect_timeout=6)
            conn = Connection(srv, user=username, password=password,
                              authentication=NTLM, auto_bind=AUTO_BIND_NO_TLS,
                              receive_timeout=10)
            _tl.c[label] = conn
        except Exception:
            _tl.c[label] = None
    return _tl.c[label]


def find_domain(sam: str, domains: dict, username: str, password: str) -> str:
    sam = sam.strip()
    if not sam:
        return "BLANK"

    filt = f"(&(objectClass=user)(sAMAccountName={sam}))"

    for label, info in domains.items():
        conn = _conn(label, info, username, password)
        if conn is None:
            continue
        try:
            conn.search(
                search_base=info["base_dn"],
                search_filter=filt,
                search_scope=SUBTREE,
                attributes=["userAccountControl"],
                time_limit=8,
                size_limit=1,
            )
            if conn.entries:
                uac = conn.entries[0].userAccountControl.value
                if isinstance(uac, int) and (uac & 0x2):
                    return f"{label} (DISABLED)"
                return label
        except Exception:
            _tl.c.pop(label, None)   # drop broken connection — will reconnect next call

    return "NOT FOUND"


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
    username = AD_USERNAME or input("AD Username (e.g. TDBFG\\your.name): ").strip()
    password = AD_PASSWORD or getpass.getpass("AD Password: ")

    # Discover
    domains = discover_domains_via_powershell()
    if not domains:
        print("\nERROR: No domains discovered. Ensure RSAT AD tools are installed or nltest is available.")
        return

    # Validate
    domains = validate_domains(domains, username, password)
    if not domains:
        print("\nERROR: Could not bind to any DC. Check credentials (format: DOMAIN\\user) and network.")
        return

    # Load data
    print(f"\n[3/3] Processing users...")
    df = pd.read_excel(INPUT_FILE, dtype=str)

    col = "TD User ID"
    if col not in df.columns:
        hit = [c for c in df.columns if c.strip().lower() == col.lower()]
        col = hit[0] if hit else None
    if not col:
        raise ValueError(f"'TD User ID' column not found. Available: {list(df.columns)}")

    user_ids = df[col].fillna("").tolist()
    total    = len(user_ids)
    print(f"  {total:,} rows  |  {len(domains)} domain(s)  |  {MAX_WORKERS} threads\n")

    results        = load_checkpoint()
    remaining      = [(i, uid) for i, uid in enumerate(user_ids) if str(i) not in results]
    done_since_save = 0
    lock           = threading.Lock()

    if results:
        print(f"  Resuming: {len(results):,} already done, {len(remaining):,} remaining.\n")

    def lookup(args):
        idx, uid = args
        return str(idx), find_domain(uid, domains, username, password)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(lookup, item): item for item in remaining}
        with tqdm(total=len(remaining), unit="user", dynamic_ncols=True) as bar:
            for f in as_completed(futures):
                idx_str, domain = f.result()
                with lock:
                    results[idx_str] = domain
                    done_since_save += 1
                    bar.set_postfix_str(f"last={domain}", refresh=False)
                    bar.update(1)
                    if done_since_save >= BATCH_SAVE_EVERY:
                        save_checkpoint(results)
                        done_since_save = 0

    save_checkpoint(results)

    # Write output
    print(f"\nWriting {OUTPUT_FILE} ...")
    df["Domain"] = [results.get(str(i), "NOT FOUND") for i in range(total)]

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Users")
        ws = writer.sheets["Users"]

        from openpyxl.styles import PatternFill
        red   = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
        green = PatternFill(start_color="CCFFCC", end_color="CCFFCC", fill_type="solid")
        dcol  = df.columns.get_loc("Domain")  # 0-based index

        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            val = row[dcol].value
            if val in ("NOT FOUND", "BLANK"):
                for cell in row:
                    cell.fill = red
            elif val and "DISABLED" not in str(val):
                row[dcol].fill = green

        for col_cells in ws.columns:
            w = max((len(str(c.value or "")) for c in col_cells), default=10)
            ws.column_dimensions[col_cells[0].column_letter].width = min(w + 4, 60)

    print(f"\nDone!  {OUTPUT_FILE}")
    print(f"Checkpoint (safe to delete): {CHECKPOINT}\n")
    print("─── Domain Breakdown ───────────────────────────")
    print(df["Domain"].value_counts().to_string())
    print("────────────────────────────────────────────────")


if __name__ == "__main__":
    main()

```