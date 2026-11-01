# Project Greenlight: Microsoft 365 Copilot Prompt Pack

**Purpose:** Build a PowerPoint outline, speaker script, demo flow and Q&A for three meetings that explain how EMS was integrated into USF.

## Before you run it

1. Use **Microsoft 365 Copilot Chat with Work mode on**. Use the Researcher agent for Prompt 1 if your account has it.
2. Replace the three items in square brackets in Prompt 1.
3. Paste one prompt at a time, and review each answer before you continue. I split the work because Copilot prompt limits vary and long prompts can be cut off.
4. If Copilot cannot see OneNote or Desktop files, move those notes to OneDrive or attach them, then rerun Prompt 1. Confirmation of what it can read is built into Prompt 1.
5. If Copilot does not create a `.pptx`, open PowerPoint, start Copilot, and paste the slide content from Prompt 4.

---

## Prompt 1: Find and inventory the sources

```text
You are helping me prepare to lead three meetings on Project Greenlight, the integration of EMS into our USF procurement intake application.

Meeting audiences:
1. [TEAM 1 AND WHAT THEY CARE ABOUT]
2. [TEAM 2 AND WHAT THEY CARE ABOUT]
3. [TEAM 3 AND WHAT THEY CARE ABOUT]

Search all work content available to you: OneNote, emails, Teams chats and transcripts, meetings, Word, PDF, PowerPoint, Excel, Loop, SharePoint, OneDrive and files saved to Desktop.

Search terms: Project Greenlight, Greenlight, USF, USFF, UIF, Unified Storefront, Unified Intake Form, EMS, Expense Management System, SPID, Signing Party, Pipeline, Vendor Expense, MLP1, MVP1, BOM, Agenda Builder, Expense Policy, Financial Approval Matrix, Approval to Hire, Management Consulting, Ariba.

Prioritize these if found: EMS_USF_MLP1_Vendor_Expense_Requirements, USFF_Pipeline_SPID_Final_Field_Mapping, USFF_EMS_Business_Rules, EMS Field Reference Rules, USFF_EMS_Implementation_Roadmap, the email threads "EMS /USF Working Sessions", "EMS + USF MLP1 Release Review & Business Clarifications" and "RECAP", the meeting "EMS System - A Quick Chat", and the BOM session minutes from Oct 22, Oct 27 and Oct 30.

Do not write the presentation yet. Return:
A. A table: source name and link, type, date, owner, what it covers, and reliability (latest decision, working note, or superseded).
B. Which locations you could search and which you could not, stated explicitly. Include OneNote and Desktop.
C. Important sources you expected but did not find.
D. Conflicts between sources, and which appears most recent.

Use only what you actually open. Do not guess at file contents.
```

---

## Prompt 2: Build the verified fact base

```text
Using the sources above, build a fact base on what has actually been implemented in USF for EMS. Separate confirmed build facts from requirements and plans. A requirement is not proof that something was built.

Cover each area where evidence exists:
- Scope: what is live or built for Vendor Expense, and what remains in EMS
- Entry and triage: GTS vs non-GTS, spend threshold, Pipeline ID validation, redirects
- Intake: forms, fields, conditional logic and lifecycle stages
- Data: how the USF schema differs from the EMS SharePoint lists, record structure, mapping, IDs, snapshots and history
- Pipeline: statuses, levels, Agenda Builder, BOM permissions, decisions and Send Back
- Management Consulting: how it is flagged and reviewed
- Finance review: Finance Partner Matrix and decisions
- Ariba: what is manual, what is automated, and which references are stored
- SPID: when it starts, field carry-forward, delta display, approval routing
- Admin dashboard: every EMS-related setting. Explain specifically how the Expense Policy is maintained and how it drives SPID approvals
- Roles, access, notifications, dashboards, exports, audit trail
- Known limitations, open decisions and BOM feedback

For every item give:
What changed from EMS | How we implemented it in USF | Where to see it in the app | Why we did it | Evidence source | Status: Confirmed built, Planned, Conflicting, or Not found

Mark anything unverified as [VERIFY]. Never invent a screen name, table name, field, number, date, person or approval rule. At the end, list the questions I need to confirm with my developers.
```

---

## Prompt 3: Create the storyboard

```text
Using only the verified fact base, create one storyboard for the shared core content and a short tailored section for each of the three teams.

Audience needs: each team should understand what changed, how it affects them, what stays the same, what they need to do, and what is still open.

Target length: 45 minutes, including about 15 to 20 minutes of live demo and 10 minutes of questions. Keep the main deck to about 10 slides.

Use this structure:
1. Why we integrated EMS into USF
2. Scope: what moved and what did not
3. End-to-end flow in USF
4. Key changes by stage
5. How the data maps from EMS to USF
6. How approvals work, including Pipeline, Agenda Builder and SPID Expense Policy
7. Demo transition
8. Impact on this team
9. Known limitations and open items
10. Next steps and decisions

For each slide provide: an action title stated as a full sentence, the one message, the recommended visual, evidence source, and any [VERIFY] items. Ask me only what would materially change the storyline.
```

---

## Prompt 4: Create the PowerPoint content

```text
Create the PowerPoint from the approved storyboard. If you can create a .pptx file, create it. If not, give me content that I can paste into Copilot in PowerPoint.

Audience: professional and executive-level teams. Style: clean, consulting-grade, one message per slide. Use a white background, dark green 1A5336, accent green 008A00, and red only for risks. Do not use cartoons, decorative icons, purple, default blue, accent lines under titles, or sidebar stripes.

For every slide:
- Full-sentence action title, under 14 words
- No more than 4 short bullets, or one table or diagram
- Native editable tables and shapes where possible
- Source reference in the notes
- [VERIFY] kept visible in the notes until I remove it

Required visuals: the end-to-end USF flow from left to right; an EMS to USF key changes table (Area | In EMS | In USF | How we did it | Impact); a Pipeline vs SPID approval comparison; and a "what moved, what stays" view.

Add an appendix for field mapping, role permissions, admin settings, status definitions, and detailed open items. Do not invent any number or timeline.
```

---

## Prompt 5: Create the script, demo flow and Q&A

```text
Write my presenter package for all three meetings in a natural, confident spoken style. Use short sentences. Do not use em dashes or buzzwords.

PART 1: SCRIPT
For each slide provide:
- What I say, word for word
- The specific implementation detail behind it
- The transition to the next slide
- Approximate speaking time

Be specific. For example, explain where the Expense Policy sits in the Admin dashboard, who maintains it, and how it affects SPID approvals, using only verified details.

PART 2: TEAM VERSIONS
For each of the three teams, list what to emphasize, what to shorten, and the exact tailored wording.

PART 3: LIVE DEMO RUNBOOK
Provide a table with: Step | Persona or login | Screen and click path | What to show | What to say | Expected result | Backup if it fails.
Use one realistic flow:
1. Start the Vendor Expense request in USF and show triage
2. Show the Pipeline requirement and intake
3. Show Pipeline status and the request drill-down
4. Log in as a BOM Lead and show Agenda Builder
5. Record decisions and show status movement and audit trail
6. Continue to the sourcing intake and show carried-forward data
7. Show Finance review
8. Show the Ariba stage and what remains manual
9. Start SPID and show final data and deltas
10. Open the Admin dashboard and show the Expense Policy for SPID approvals
11. Show dashboards, exports and what remains in EMS
Mark any step you cannot verify with [VERIFY]. Add a pre-demo checklist for test users, seed data, and browser tabs.

PART 4: Q&A PREP
Write at least 25 likely questions and concise answers, grouped by business, BOM, finance, procurement, technical, data, security, audit and support. Include why we changed something, how data maps from SharePoint, what happens to historic EMS records, how approvals are determined, what happens when a request gets stuck, and what remains out of scope. Cite the source. If unknown, write: "I do not want to guess. I will confirm with [owner] and follow up."

PART 5: FINAL CHECK
List every [VERIFY], who can confirm it, and what must be checked in the app before the first meeting.
```

---

## Final check before you present

- Click through the demo in the current environment. Copilot can identify intended behaviour from documents, but it cannot prove what the current build does.
- Clear every `[VERIFY]` with your developers.
- Check source links and remove anything your audience should not see.
