# AD Group Reconciliation Tool
## Enterprise Setup & Usage Guide

---

## Folder Structure

```
ad_reconciliation/
├── ad_reconciliation.py   ← Main script
├── requirements.txt       ← Python dependencies
├── README.md              ← This file
├── input/                 ← Drop your monthly Excel roster here
├── reports/               ← Generated reconciliation reports land here
├── archive/               ← Previous reports are auto-archived here
└── logs/                  ← Execution logs
```

---

## One-Time Setup

```bat
:: 1. Install Python 3.11+ from https://python.org (check "Add to PATH")

:: 2. Open Command Prompt in the ad_reconciliation folder
cd C:\path\to\ad_reconciliation

:: 3. (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

:: 4. Install dependencies
pip install -r requirements.txt
```

---

## Configuration (REQUIRED before first run)

Open `ad_reconciliation.py` and edit the `CONFIG` block at the top.

### Minimum required changes:

| Key | What to set |
|---|---|
| `AD_GROUP_NAME` | Exact name of your AD group (e.g. `"GRP-BANKING-USERS"`) |
| `EXCEL_COLUMN_USERID` | Exact column header in your monthly Excel file for user IDs |
| `MATCH_AD_FIELD` | Which AD field to compare on (`AD_COLUMN_SAMACCOUNTNAME` is most common) |
| `MATCH_EXCEL_FIELD` | Which Excel field to compare on (`EXCEL_COLUMN_USERID` matches the above) |

### Common match scenarios:

**Scenario A – Match on Windows login name vs a "Username" column in Excel:**
```python
"AD_COLUMN_SAMACCOUNTNAME": "SamAccountName",
"EXCEL_COLUMN_USERID":      "Username",        # ← your Excel header
"MATCH_AD_FIELD":    "AD_COLUMN_SAMACCOUNTNAME",
"MATCH_EXCEL_FIELD": "EXCEL_COLUMN_USERID",
```

**Scenario B – Match on Employee ID:**
```python
"AD_COLUMN_EMPLOYEEID":     "EmployeeID",
"EXCEL_COLUMN_EMPLOYEEID":  "EmpID",           # ← your Excel header
"MATCH_AD_FIELD":    "AD_COLUMN_EMPLOYEEID",
"MATCH_EXCEL_FIELD": "EXCEL_COLUMN_EMPLOYEEID",
```

**Scenario C – Match on email address:**
```python
"AD_COLUMN_MAIL":       "Mail",
"EXCEL_COLUMN_EMAIL":   "EmailAddress",        # ← your Excel header
"MATCH_AD_FIELD":    "AD_COLUMN_MAIL",
"MATCH_EXCEL_FIELD": "EXCEL_COLUMN_EMAIL",
```

---

## Monthly Execution (Every Month)

```bat
:: Step 1 – Drop this month's Excel roster into the input/ folder
:: Step 2 – Run the script
cd C:\path\to\ad_reconciliation
.venv\Scripts\activate
python ad_reconciliation.py

:: The report opens in reports/
```

---

## Dry Run (test without writing files)

```python
# In CONFIG, set:
"DRY_RUN": True,
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No Excel files found in 'input/'` | Drop your .xlsx monthly file into the `input/` sub-folder |
| `Required column 'UserID' not found` | Check the exact spelling of your Excel column header and update `EXCEL_COLUMN_USERID` in CONFIG |
| `PowerShell exited with code 1` | Run `Get-ADGroupMember -Identity "GROUP_NAME"` in PowerShell manually to confirm you have permissions |
| `AD query exceeded timeout` | Increase `AD_QUERY_TIMEOUT_SECONDS` (try 600 for large groups) |
| `Group may be empty or name is wrong` | Verify the group name in AD Users & Computers — it must be the **pre-Windows 2000 name** (SAM name) |
| Matched count is 0 but you expect matches | Your column names may not align. Enable `"LOG_LEVEL": "DEBUG"` and check the logged column names |

---

## Output Tabs Explained

| Tab | Contents | Colour |
|---|---|---|
| **Summary** | Stats, timestamps, config used | Blue |
| **Users_To_Add** | In Excel roster but NOT in AD → action required | Green |
| **Users_To_Remove** | In AD but NOT in Excel roster → action required | Red |
| **Matching_Users** | Found in both — no action needed | Blue |
| **Duplicate_Records** | Records with duplicate key values | Yellow |
| **Invalid_Records** | Rows where the match key is blank/null | Pink |
| **Raw_AD_Output** | Every record returned by AD query | Grey |
