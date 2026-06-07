# Claude Document Creation Skills Guide
# For use with GitHub Copilot (Claude Sonnet / GPT) inside VS Code on Windows

---

## HOW THIS FILE WORKS

This file lives in ONE place on your Windows machine and applies to EVERY VS Code
window you open — no per-project setup needed.

Where to save it:
  C:\Users\<YourUsername>\AppData\Roaming\Code\User\prompts\td_bank_skills.md

The global Copilot instructions file (see Part 4) tells Copilot to always read
this file before creating any document. Because it is set globally in VS Code
User Settings, it applies to every workspace and every VS Code window automatically.

---

## TD BANK BRAND CONSTANTS

Always apply these values to every document without exception.

### Colors

| Role              | Name              | Hex       |
|-------------------|-------------------|-----------|
| Primary           | TD Green          | #00A651   |
| Dark              | TD Dark Green     | #005427   |
| Accent            | TD Gold           | #FFD700   |
| Background        | White             | #FFFFFF   |
| Background alt    | Light Grey        | #F5F5F5   |
| Text primary      | Near-Black        | #1A1A1A   |
| Text secondary    | Grey              | #555555   |
| Table header fill | TD Green          | #00A651   |
| Table alt row     | Light Green Tint  | #E8F5E9   |
| Borders           | Medium Grey       | #CCCCCC   |

### Typography

| Element          | Font        | Size    | Style                     |
|------------------|-------------|---------|---------------------------|
| Document Title   | Arial       | 28–32pt | Bold                      |
| Heading 1        | Arial       | 22pt    | Bold, TD Green #00A651    |
| Heading 2        | Arial       | 18pt    | Bold, TD Dark Green       |
| Body Text        | Arial       | 11pt    | Regular, #1A1A1A          |
| Captions         | Arial       | 9pt     | Italic, Grey              |
| Table Headers    | Arial       | 11pt    | Bold, White on TD Green   |

---

## PART 1 — WORD DOCUMENTS (.docx)

Claude/Copilot generates .docx files using the `docx` npm package in JavaScript/Node.js.

### Critical Rules — Read Before Every Word Doc

- NEVER use \n for line breaks — use separate Paragraph() elements
- NEVER use unicode bullets (•) — use the docx numbering/LevelFormat.BULLET system
- ALWAYS set page size explicitly to US Letter (12240 x 15840 twips) — default is A4
- ALWAYS use WidthType.DXA for table widths — PERCENTAGE breaks in Word/Google Docs
- ALWAYS set width on both the Table AND each individual Cell
- ALWAYS use ShadingType.CLEAR for cell backgrounds — ShadingType.SOLID causes black fills
- ALWAYS set columnWidths on the table definition

### TD Bank Word — JavaScript Config Block

Paste this into your prompt every time you ask for a .docx:

```javascript
const TD = {
  colors: {
    green:      "00A651",
    darkGreen:  "005427",
    gold:       "FFD700",
    white:      "FFFFFF",
    lightGrey:  "F5F5F5",
    textPrimary:"1A1A1A",
    textMuted:  "555555",
    tableAlt:   "E8F5E9",
    border:     "CCCCCC",
  },
  font: "Arial",
  sizes: {        // docx half-points
    title: 56,    // 28pt
    h1:    44,    // 22pt
    h2:    36,    // 18pt
    body:  22,    // 11pt
    small: 18,    // 9pt
  },
  page: {
    width:  12240,  // US Letter
    height: 15840,
    margin: 1440,   // 1 inch
  }
};
```

### TD Bank Word — Document Structure

Cover Page
  TD Green full-width bar across top (80pt tall, white "TD Bank" text)
  Document title — Arial Bold 28pt near-black, centered, vertically at 40%
  Subtitle / date — Arial 14pt grey, centered below title
  TD Green full-width bar across bottom

Header (every page after cover)
  Left:   "TD Bank" in TD Green bold 9pt
  Center: Document title 9pt grey
  Right:  Page number 9pt grey
  Bottom border: TD Green 2pt rule

Body
  H1 — Arial Bold 22pt TD Green (#00A651), spacing before 240 after 120
  H2 — Arial Bold 18pt TD Dark Green (#005427), spacing before 180 after 80
  Body — Arial 11pt #1A1A1A, line spacing 1.15

Tables
  Header row: fill #00A651 (ShadingType.CLEAR), text white Arial Bold 11pt
  Alternating rows: white / #E8F5E9 (ShadingType.CLEAR)
  All borders: #CCCCCC 1pt
  Width: WidthType.DXA set on table AND each cell

Footer (every page)
  Left:  "© TD Bank Group — Confidential" 9pt grey
  Right: "Page N of M" 9pt grey
  Top border: TD Green 1pt rule

### Example Prompt — Word Doc

Copy-paste this into Copilot chat, fill in your content:

---
You are a document specialist. Create a professional Word document (.docx) for TD Bank.

THEME (apply exactly):
- Page: US Letter (12240x15840 twips), 1440 twip margins (1 inch)
- Font: Arial throughout
- TD Green: #00A651 | Dark Green: #005427 | Table alt row: #E8F5E9
- H1: Arial Bold 22pt TD Green | H2: Arial Bold 18pt dark green | Body: Arial 11pt #1A1A1A
- Cover: green top bar + white text "TD Bank", centered title, green bottom bar
- Header: "TD Bank" green left | doc title center | page # right | green bottom border
- Footer: "© TD Bank Group — Confidential" left | "Page N of M" right | green top border
- Tables: header row green fill (ShadingType.CLEAR) white text | alt rows #E8F5E9 | grey borders

RULES:
- Use separate Paragraph() for every line break — never \n
- WidthType.DXA for ALL table widths (table + cells)
- ShadingType.CLEAR for ALL cell backgrounds
- Set columnWidths on every table

CONTENT:
[paste your content here]

Save to: C:\Users\<YourUsername>\Documents\TD_Output\[filename].docx
---

---

## PART 2 — PDF DOCUMENTS

Claude/Copilot generates PDFs using Python's reportlab library.

### Critical Rules — Read Before Every PDF

- reportlab colors are 0.0–1.0 float tuples, NOT hex strings
- Use Helvetica as the safe Windows substitute for Arial (always available)
- Use letter page size from reportlab.lib.pagesizes (612 x 792 points)
- 1 inch = 72 points in reportlab
- Page numbers require a PageTemplate with onPage callback — not inline
- Use platypus.Table with TableStyle for data tables
- Never use Unicode subscripts directly — use Paragraph XML tags

### TD Bank PDF — Python Config Block

```python
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# Page
PAGE_W, PAGE_H = letter   # 612 x 792
MARGIN = 1 * inch          # 72 points

# TD Bank colors (0.0–1.0 float tuples)
TD = {
    "green":      colors.Color(0/255, 166/255,  81/255),   # #00A651
    "dark_green": colors.Color(0/255,  84/255,  39/255),   # #005427
    "gold":       colors.Color(255/255, 215/255,  0/255),  # #FFD700
    "white":      colors.white,
    "light_grey": colors.Color(0.96, 0.96, 0.96),
    "text":       colors.Color(0.10, 0.10, 0.10),
    "text_muted": colors.Color(0.33, 0.33, 0.33),
    "table_alt":  colors.Color(0.91, 0.96, 0.91),          # #E8F5E9
    "border":     colors.Color(0.80, 0.80, 0.80),
}

FONTS = {
    "heading": "Helvetica-Bold",
    "body":    "Helvetica",
    "small":   "Helvetica-Oblique",
}
```

### TD Bank PDF — Document Structure

Cover Page
  Full-width green banner top (80pt): "TD Bank" white Helvetica-Bold 28pt
  Title: Helvetica-Bold 24pt #1A1A1A, centered, y=420
  Subtitle/date: Helvetica 14pt grey, centered, y=395
  Full-width green bar bottom (40pt)

Body Pages (header + footer drawn via onPage callback)
  Header: 2pt green line at y=762 | "TD Bank | [title]" Helvetica 9pt grey below it
  H1: Helvetica-Bold 18pt green | 2pt green underline rule below
  H2: Helvetica-Bold 14pt dark green
  Body: Helvetica 11pt #1A1A1A, leading=16
  Tables: green header row | alt rows table_alt/white | 0.5pt grey borders
  Footer: grey rule at y=54 | "© TD Bank Group — Confidential" left | "Page N of M" right

### Example Prompt — PDF

---
You are a document specialist. Create a PDF for TD Bank using Python reportlab.

THEME (apply exactly):
- Page: letter (612x792), 1 inch (72pt) margins
- Fonts: Helvetica-Bold for headings, Helvetica for body
- TD Green: Color(0, 166/255, 81/255) | Dark Green: Color(0, 84/255, 39/255)
- Cover: full-width green top banner (80pt, white title text), green bottom bar (40pt)
- Header every page: 2pt green rule + "TD Bank | [doc title]" 9pt grey
- Footer every page: grey rule + "© TD Bank Group — Confidential" left + "Page N of M" right
- H1: Helvetica-Bold 18pt green with 2pt green underline
- H2: Helvetica-Bold 14pt dark green
- Body: Helvetica 11pt #1A1A1A leading=16
- Tables: green header row (white Helvetica-Bold), alternating Color(0.91,0.96,0.91)/white rows

CONTENT:
[paste your content here]

Save to: C:\Users\<YourUsername>\Documents\TD_Output\[filename].pdf
---

---

## PART 3 — POWERPOINT (.pptx)

Claude/Copilot generates .pptx files using the pptxgenjs npm library.

### TD Bank PowerPoint — JavaScript Config Block

```javascript
const TD_THEME = {
  colors: {
    green:     "00A651",
    darkGreen: "005427",
    gold:      "FFD700",
    white:     "FFFFFF",
    text:      "1A1A1A",
    textMuted: "777777",
    tableAlt:  "E8F5E9",
  },
  fonts: {
    title:   "Arial Black",
    heading: "Arial",
    body:    "Arial",
  },
  slide: { w: 13.33, h: 7.5 },  // 16:9 widescreen, inches
};
```

### TD Bank Slide Layouts

TITLE SLIDE
  Background: TD Dark Green (#005427)
  Left accent bar: TD Green (#00A651), w=0.4", full height, x=0
  Title: white Arial Black 36pt, x=0.7 y=2.8 w=12 h=1.2
  Subtitle: TD Gold (#FFD700) Arial 18pt, x=0.7 y=4.1
  Footer: "© TD Bank Group" white Arial 9pt, x=11 y=7.1

CONTENT SLIDE
  Background: white
  Top bar: TD Green full width, h=0.7", y=0 — contains slide title white Arial Bold 24pt
  Content area: starts y=0.9, left/right margin 0.4"
  Body text: Arial 14pt #1A1A1A
  Footer: "TD Bank | Confidential" grey 9pt bottom-left | page number bottom-right

SECTION DIVIDER
  Background: TD Green (#00A651) full bleed
  Section number: white Arial 48pt light, x=0.6 y=2.8
  Section title: white Arial Black 32pt, x=0.6 y=3.8

TWO-COLUMN SLIDE
  Green top bar (same as content slide)
  Left 50%: text/bullets
  Right 50%: image, chart, or stat callout box

DATA TABLE SLIDE
  Green top bar with title
  Table: green header row (white text) | alt rows #E8F5E9/white | grey 0.5pt borders
  Totals row (if present): dark green fill, white bold text

CLOSING SLIDE
  Background: TD Dark Green (#005427)
  Main text: white Arial Black 40pt centered, y=2.8
  Sub text: white Arial 18pt centered, y=4.1

### Design Rules — Non-Negotiable

- Green dominates: 60–70% visual weight per slide
- Font sizes: Titles 36–44pt | Section headers 24pt | Body 14–16pt | Captions 10pt
- Minimum margins: 0.4" from all slide edges
- Every slide must have at least one visual element (chart, icon, callout, image)
- No pure text slides
- White text only on green/dark green backgrounds — check contrast always

### Example Prompt — PowerPoint

---
You are a presentation specialist. Create a PowerPoint (.pptx) for TD Bank using pptxgenjs.

THEME (apply exactly):
- Slide size: 13.33 x 7.5 inches (16:9)
- Title slide: dark green (#005427) bg, TD green left accent bar (0.4" wide), white Arial Black title, gold subtitle
- Content slides: white bg, TD green top bar (0.7" tall, full width) with white Arial Bold 24pt title
- Section dividers: solid TD green (#00A651) bg, white Arial Black title
- Tables: green (#00A651) header row (white text), alternating #E8F5E9/white rows, grey 0.5pt borders
- Footer every slide: "TD Bank | Confidential" grey 9pt bottom-left | page number bottom-right
- Fonts: Arial Black for titles, Arial for all body text

SLIDES:
1. Title: [your title]
[describe each slide]

Save to: C:\Users\<YourUsername>\Documents\TD_Output\[filename].pptx
---

---

## PART 4 — VS CODE GLOBAL SETUP (WINDOWS, EVERY WINDOW)

This makes the TD Bank theme apply automatically in EVERY VS Code window — no
per-project setup, no repeating yourself.

### Step 1 — Create Your Output Folder

Open File Explorer and create:
  C:\Users\<YourUsername>\Documents\TD_Output\

This is where all generated documents will be saved.

### Step 2 — Save This Skills File Globally

Save this file to:
  C:\Users\<YourUsername>\AppData\Roaming\Code\User\prompts\td_bank_skills.md

To get there quickly:
  Press Win + R → type %APPDATA%\Code\User\ → press Enter
  Create a new folder called "prompts" if it doesn't exist
  Save this file inside it as td_bank_skills.md

### Step 3 — Set Global Copilot Instructions in VS Code

This is the key step that makes it apply to EVERY VS Code window.

  1. Open VS Code
  2. Press Ctrl + Shift + P (Command Palette)
  3. Type: "Open User Settings (JSON)" — select it
  4. Add this block inside the outermost { } of your settings.json:

    "github.copilot.chat.codeGeneration.instructions": [
      {
        "file": "${env:APPDATA}\\Code\\User\\prompts\\td_bank_skills.md"
      }
    ]

  5. Save the file (Ctrl + S)

This tells GitHub Copilot to automatically load your TD Bank skills file as
background context in every chat session, in every VS Code window, globally.

### Step 4 — Install Required Libraries (One Time)

Open any VS Code terminal (Ctrl + `) and run:

  For Word and PowerPoint:
    npm install -g docx pptxgenjs

  For PDF:
    pip install reportlab

  Verify Node is installed:
    node --version

  If node is not found, download from https://nodejs.org (LTS version).
  If pip is not found, Python may need to be installed or added to PATH.

### Step 5 — Enable Copilot Agent Mode

In the Copilot chat panel (the speech bubble icon in the left sidebar):
  Click the dropdown at the top of the chat panel
  Select "Agent" mode (not "Ask" or "Edit")

Agent mode allows Copilot to write AND run the code to generate your files.
In Ask mode it will only write code — you'd have to run it yourself.

### Step 6 — Set Your Default Model to Claude Sonnet

In the Copilot chat panel:
  Click the model selector (usually shows current model name at the bottom)
  Select "Claude Sonnet 4.5" or the latest Claude Sonnet available

Claude Sonnet produces significantly better document code than GPT for this
use case. Use GPT as a fallback only.

---

## PART 5 — DAILY USAGE IN PRACTICE

### Starting a Document Session

1. Open VS Code (any project, any folder — settings are global)
2. Open Copilot Chat: Ctrl + Alt + I (or click the chat icon in the sidebar)
3. Make sure you are in Agent mode and Claude Sonnet is selected
4. Copy the relevant example prompt from Part 1, 2, or 3 above
5. Fill in your content and file name
6. Send — Copilot will write the code, run it, and save the file

### Verifying the Output

After Copilot finishes, check:
  C:\Users\<YourUsername>\Documents\TD_Output\

Open the file in Word / Adobe / PowerPoint and verify:
  ✓ TD Green headers and table rows
  ✓ Arial font throughout
  ✓ Correct cover page layout
  ✓ Header and footer on every page

### If Something Looks Wrong

Add any of these corrections to your next message in the same chat:

  "The table header row is showing as black — fix the shading to use ShadingType.CLEAR"
  "The heading colour is wrong — change to #00A651"
  "The page margins are too wide — set to 1440 twips (1 inch) on all sides"
  "The footer is missing — add it to every page using a Footer section"
  "The font is not Arial — change all fonts to Arial"

### Saving Prompts You Use Often

Create a file at:
  C:\Users\<YourUsername>\AppData\Roaming\Code\User\prompts\my_td_prompts.md

Paste your most-used prompts in there. Then from Copilot chat you can say:
  "Use my prompt template from my_td_prompts.md for a client summary report"

---

## QUICK REFERENCE

| Document type     | Library     | Language   | Command to install        |
|-------------------|-------------|------------|---------------------------|
| Word (.docx)      | docx        | JavaScript | npm install -g docx       |
| PowerPoint (.pptx)| pptxgenjs   | JavaScript | npm install -g pptxgenjs  |
| PDF               | reportlab   | Python     | pip install reportlab      |

| Color name       | Hex     | JS string  | Python float tuple              |
|------------------|---------|------------|---------------------------------|
| TD Green         | #00A651 | "00A651"   | Color(0, 166/255, 81/255)       |
| TD Dark Green    | #005427 | "005427"   | Color(0, 84/255, 39/255)        |
| TD Gold          | #FFD700 | "FFD700"   | Color(255/255, 215/255, 0)      |
| Table alt row    | #E8F5E9 | "E8F5E9"   | Color(0.91, 0.96, 0.91)         |

---

## TROUBLESHOOTING

| Problem                        | Fix                                                      |
|--------------------------------|----------------------------------------------------------|
| Copilot ignores TD theme       | Check settings.json — confirm the file path is correct   |
| Table cells appear black       | Use ShadingType.CLEAR not ShadingType.SOLID              |
| Wrong page size (A4 not Letter)| Set width:12240 height:15840 explicitly in page setup    |
| File not saved to TD_Output    | Include full path in prompt: C:\Users\...\TD_Output\     |
| node not found                 | Install Node.js from nodejs.org, restart VS Code         |
| pip not found                  | Run: python -m pip install reportlab                     |
| Agent mode not available       | Update VS Code and GitHub Copilot extension              |
| Claude not listed as model     | Your Copilot license may need Claude access enabled      |

---

Last updated: June 2026
TD Bank Document Creation Skills Guide — Global VS Code Edition
