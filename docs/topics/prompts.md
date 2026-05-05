# Useful Prompts for Day-to-Day Use

A curated collection of ready-to-use prompts for productivity, analysis, communication, and more.

---

## Table of Contents

- [Email & Communication](#email--communication)
- [Analysis & Strategy](#analysis--strategy)
- [Writing & Content](#writing--content)
- [Meetings & Planning](#meetings--planning)
- [Data & Reporting](#data--reporting)
- [Technical Documentation](#technical-documentation)
- [Personal Productivity](#personal-productivity)
- [Knowledge Management](#knowledge-management)

---

## Email & Communication

### Find Action Items (Last 2 Weeks in Outlook)

> Review recent emails and surface all outstanding tasks, deadlines, and follow-ups in priority order.

```text
Review my Outlook emails from the past 14 days (including Inbox and Sent Items). 
Identify all work items, action items, deliverables, follow-ups, approvals, or tasks 
that require my attention — including:

- Emails directly addressed to me (To: field)
- Emails where I am CC'd in threads with fewer than 10 participants
- Threads where I am asked to review, approve, provide input, submit documents, 
  complete tasks, or follow up

For each identified work item:
- Clearly summarize the task in one sentence
- Identify the requester
- Extract and highlight any mentioned deadlines, due dates, meetings, or key dates
- Note whether the deadline is explicit or implied
- Indicate whether the item appears completed (based on my reply) or still pending

Then:
- Remove duplicate tasks from long threads
- Group related items together
- Flag any overdue items
- Flag any items due within the next 7 days

Finally, prioritize all pending items from highest to lowest priority using the 
following logic:

1. Urgent deadline (within 3 days or overdue)
2. High business impact or blocking others
3. Requested by leadership or multiple stakeholders
4. Medium priority (due within 1–2 weeks)
5. Low priority (no deadline or informational follow-up)

Present the final output in this format:

**High Priority**
- Task:
- Requested by:
- Deadline:
- Status:
- Context:

**Medium Priority** (same format)

**Low Priority** (same format)

At the end, provide:
- A short executive summary (3–5 bullets)
- A recommended "Top 5 Things to Do Today"
```

---

### Catch Up After Vacation

> Get a full briefing on everything missed — emails, messages, and outstanding actions.

```text
Give me a complete catch up on everything I have missed. Summarise my key unread 
emails, Teams messages and any outstanding actions across all threads. Prioritise 
anything from my manager /[name]. Highlight every @mention, list anything that needs 
my response, and flag items that are overdue or time sensitive. For each email or 
thread, give me a clear one line summary, the key decisions or updates, and the next 
step I need to take. Present everything as sharp, scannable bullet points so I can 
see my priorities at a glance.
```

---

### Follow Up on Unanswered Requests

> Find every email you sent that hasn't received a reply in the last 3 business days.

```text
Identify every email request I have sent in Outlook that has not received a reply 
in the past three business days. For each one, list the subject line, who I emailed, 
and when it was sent. Add one or two sentences of context so I can remember what I 
asked for, then present the key details and required follow up in sharp bullet points 
so I can review and act quickly.
```

---

### Sound Like Yourself (Writing Style Guide)

> Analyse your sent emails to build a personalised style guide.

```text
I want you to analyze my writing style by looking at my sent emails in Outlook and 
my messages in Microsoft Teams. Go through as many as you can access and identify 
patterns in how I write — things like my tone, sentence length, how formal or casual 
I am, punctuation habits, common phrases I use, how I open and close messages, and 
anything else that makes my writing sound like me.

Once you've done that, summarize the style you've identified so I can confirm it 
sounds right. From that point on, whenever I ask you to draft a reply or write a 
message for me, use MY style — not a generic AI tone. It should sound like I wrote 
it myself.
```

---

### CoPilot Style Guide Generator

> Extract what makes your writing unique so CoPilot can replicate it.

```text
Analyse my recent emails to identify my distinct writing style and communication 
voice. Create a short copy style guide CoPilot can follow so future emails sound 
like me, covering tone, formality, cadence, language preferences, and how human 
or conversational my writing is.

Summarise this as clear, reusable instructions that preserve my natural voice rather 
than sounding generic or automated.
```

---

## Analysis & Strategy

### Extract Strategic Insights

> Analyse content like a strategy consultant and surface actionable opportunities.

```text
Analyze this text like a strategy consultant.
Identify the key ideas, missed opportunities, 
and strategic implications I should act on immediately.
```

---

### Extract What Others Overlook

> Surface hidden assumptions, biases, and blind spots experts would notice.

```text
Read this text and point out the hidden assumptions, biases or unspoken perceptions 
that most readers would overlook — but experts would notice.

Read this like an expert. Expose the assumptions, blind spots, or hidden gems that 
most people would miss.
```

---

### Compare Competing Ideas

> Contrast an argument with opposing views — where they align, clash, and why it matters.

```text
Contrast this argument with opposing views in the same field.
Show where they align, where they clash, and why it matters.
```

---

### War Room — Unstick a Problem

> Get a structured breakdown of a complex situation with honest tradeoffs and a recommended path.

```text
I'm stuck and it's costing me time. Here's the situation: [paste full context 
including what you tried]

War-room this with me:

1. What am I not seeing? (blind spots, assumptions, missing data)
2. Give me 3 approaches: Fast/Dirty, Balanced, Perfect — with honest tradeoffs
3. Which approach would YOU bet on and why? (Don't be diplomatic, pick one)
4. What's the likely failure point of your recommended approach and how do I prevent it?
5. If this still doesn't work, what's Plan B?

Think like a consultant who gets fired if I fail.
```

---

### Strategic Download — Master a Topic Fast

> Get a rapid expert-level briefing on any topic, structured for decision-making.

```text
I need to master [topic] fast enough to make decisions about it by the end of the 
week. Don't just explain it, download it into my brain. Give me:

- The 3 mental models experts use to think about this (with examples from my industry)
- The one metric that matters most and why
- The common mistake that destroys 80% of projects in this area
- A decision framework: [Situation A] = [do this], [Situation B] = [do that]
- Three questions to ask that make me sound expert-level in meetings

Make this tactical, not theoretical.
```

---

### Crystal Ball — Pre-Meeting Intelligence Briefing

> Walk into any meeting looking like you've been paying perfect attention.

```text
I've got [meeting type] with [person name] tomorrow and I need to walk in looking 
psychic. Intelligence briefing required:

1. Scan ALL our interactions (emails, chats, shared docs, meeting notes) from the 
   last 60 days
2. What are the top 5 things on their mind right now? (Show me evidence — quote 
   the emails/messages)
3. What 3 things will they ask me about? (based on their patterns and our 
   outstanding items)
4. What's the ONE thing they're worried about but haven't said directly? 
   (Read between the lines)
5. Draft 3 pre-emptive responses I can have ready

Make me look like I've been paying perfect attention.
```

---

## Writing & Content

### Summarize Text

> Get the most important points from any long text, fast.

```text
Here is a long text: [paste]. Summarize it in seconds, keeping only the most 
important and useful points.
```

---

### Executive Summary

> Convert a report into a sharp, under-200-word summary with recommendations.

```text
As a project manager, summarize the key findings of this report in under 200 words, 
including at least three practical recommendations.
```

---

### Brief Summary

> Get a list-style summary of objectives, strategies, and challenges in under 100 words.

```text
Provide a list-style summary of the following document, outlining the main objectives, 
proposed strategies, and potential challenges in under 100 words.
```

---

### Fix Writing Style

> Clean up and polish any text — clearer, smoother, and stronger.

```text
Here's my text: [paste]. Make it clearer, smoother, and stronger. Show me exactly 
what to change and why. Make the text clear, concise, and easy to read. Keep the 
original meaning, remove any confusion, and make the sentences flow naturally. 
Only return the polished final version.
```

---

### Rewrite for Persuasion

> Transform content with a stronger hook, emotional storytelling, and a clear call to action.

```text
Rewrite this content using persuasive copywriting techniques: stronger hook, 
emotional storytelling, and clear calls to action.
```

---

### Turn Ideas into Action

> Convert any material into a concrete, actionable step-by-step plan.

```text
Turn this material into a step-by-step action plan I can apply in my business 
or workflow today.
```

---

### Problem Solver

> Get a direct, expert-guided walkthrough of any problem.

```text
You're an expert in [field]. Walk me through how to solve [problem] with a direct, 
step-by-step approach.
```

---

### Prompt Generator

> Generate a ready-to-use, powerful prompt for any specific goal.

```text
Create a ready-to-use prompt that helps me achieve [specific goal]. Make it short, 
powerful, and actionable.
```

---

## Meetings & Planning

### Get Ready for a Meeting in Seconds

> Get a full briefing on a person before your next meeting.

```text
Prepare me for my upcoming meeting with /[Name]. Pull together everything relevant, 
including recent emails, Teams messages and shared documents between us. Summarise 
the key discussions, decisions, open issues and risks. Highlight every action item 
linked to this relationship, including what I owe, what they owe, and anything 
overdue or time sensitive. Give me a tight, scannable briefing with clear bullet 
points so I walk in knowing the history, the context and the next steps.
```

---

### CoPilot Executive Assistant

> Set up CoPilot to act as a senior executive assistant across all your tools.

```text
You are my executive assistant with more than ten years of experience supporting 
senior leaders. Use the information available from Outlook, Teams, SharePoint and 
any relevant internal files. Help me stay organised and focused by reviewing 
everything on my plate and identifying what matters most. Pull out my priorities, 
action items, deadlines, risks, and anything that needs a follow up. Structure it 
clearly so I know exactly what to do next and what needs my attention today.
```

---

### Execution Engine

> Turn a goal and a deadline into a clear execution plan — with tasks you can delegate, automate, or do yourself.

```text
I need to [specific goal] by [deadline]. Here's the mess I'm dealing with: 
[paste and attach everything]

Give me:
1. Step-by-step execution plan with time estimates for each step
2. Which steps YOU can do right now (drafting, analysis, research)
3. Which steps I should delegate or automate (name the specific tools)
4. Which steps only I can do (decisions, relationships, approvals)
5. The 3 biggest risks that will derail this and how to prevent them

Then execute step 1 for me immediately.
```

---

## Data & Reporting

### Excel — Find the Story in the Numbers

> Analyse a spreadsheet like a consultant briefing a busy executive.

```text
Analyse this Excel file as if you're explaining it to a busy executive who has 
5 minutes. I need you to identify:

1. The three strongest patterns or trends you see in the data. Tell me what they 
   are, where you found them, and why they matter.
2. The three biggest risks or warning signs. Explain what could go wrong and how 
   urgent each risk is.
3. Any data points that look unusual or out of place. Distinguish between actual 
   errors and genuine outliers that need investigation.

Present your findings in plain English with no jargon. For each pattern, risk, or 
anomaly, include the specific location in the spreadsheet (sheet name, column/row 
numbers) and actual figures as evidence. End with 2–3 concrete recommendations for 
what I should do next based on what you found.
```

---

### Create the Summary an Exec Actually Wants

> Turn a spreadsheet into a one-page, decision-ready executive summary.

```text
Turn this spreadsheet into a one-page executive summary that fits the following 
structure:

- Start with one headline sentence capturing the most important finding
- Top 3 trends (what's happening, whether it's improving or declining, and why 
  it matters)
- Top 3 outliers or unusual data points (what's off, how significant, and your 
  best guess why)
- Performance snapshot: what's doing best, what's doing worst, and what changed 
  the most

End with three prioritized actions:
- One thing to do immediately this week
- One thing to do this month
- One thing that needs further investigation before we can act

Keep the entire summary under 400 words, avoid jargon, and make sure every figure 
has context (e.g. "down 15% compared to last quarter" instead of just "15% down"). 
If you are not confident about something, say so clearly.
```

---

## Technical Documentation

### Executive Briefing for Leadership

> Convert messy context into a concise, executive-ready leadership briefing.

```text
Act as a senior program manager preparing an executive-level briefing for senior 
leadership.

Your task is to convert the information I provide into a concise, executive-ready 
summary suitable for managers and directors.

Follow these strict guidelines:
- Be concise and factual
- Use clear executive language with no technical jargon unless necessary
- Focus on impact, timeline, and decisions required
- Avoid unnecessary background details
- Use short sections and bullet points
- The tone should be calm, confident, and solution-oriented

Structure the response exactly as follows:

**1. Executive Summary**
2–3 sentence overview explaining the situation and why leadership should be aware.

**2. Current Situation**
What has changed and why (1–3 bullet points).

**3. Impact**
Business or delivery impact (timeline, dependencies, stakeholders).

**4. Proposed Path Forward**
Clear action steps being taken to address the situation.

**5. Leadership Awareness / Decision (if applicable)**
State whether leadership action, approval, or simply awareness is required.

Keep the entire response under 200 words. Prioritize clarity and executive 
readability.

Here is the context:
[PASTE CONTEXT HERE]
```

---

### UiPath CoE Deployment Documentation

> Generate complete Confluence-ready deployment documentation for a UiPath Center of Excellence.

```text
Act as a senior DevOps architect and technical documentation expert responsible for 
creating onboarding documentation for a UiPath Center of Excellence (CoE).

Your task is to generate a clear and visually structured deployment process for 
UiPath automations moving from Development to Production.

The output should be formatted as a Confluence documentation page that includes:

1. A high-level deployment flowchart
2. A clear breakdown of each stage in the pipeline
3. Approval gates and responsible roles
4. Environment progression
5. Link placeholders for documentation, pipelines, and approvals
6. Expandable sections for deeper explanations
7. A diagram structure that can easily be recreated using draw.io inside Confluence

Structure the output as follows:

**SECTION 1 — Executive Overview**
Short explanation of how UiPath deployments work and the purpose of the pipeline.

**SECTION 2 — Deployment Flow Diagram**
Full lifecycle flowchart including:
- Development Environment → Code Review → Package Creation → Orchestrator 
  Deployment → UAT → Business Approval → Production Release
- Decision points, approval gates, feedback loops, and responsible roles

Also provide a Mermaid diagram version.

**SECTION 3 — Detailed Phase Breakdown**
For each stage: Stage Name, Purpose, Responsible Role, Inputs, Actions, Approval 
Required, Outputs / Deliverables.

**SECTION 4 — Approval Workflow**
Table showing: Stage | Approver | Approval Tool | Validation Criteria.

**SECTION 5 — Environment Promotion Model**
DEV → TEST/UAT → PROD with promotion rules for each environment.

**SECTION 6 — Expandable Operational Details**
- Developer Deployment Steps
- UAT Validation Process
- Production Deployment Checklist
- Rollback Process

**SECTION 7 — Quick Visual Summary**
Simplified step-by-step reference for new engineers.

After this prompt I will provide the exact deployment steps, approvals, and tools 
used in our organization.
```

---

### UiPath Deployment Flowchart (draw.io)

> Generate a structured deployment flowchart blueprint ready for draw.io / Confluence.

```text
Act as a senior DevOps architect designing a UiPath deployment flowchart for draw.io.

The diagram must include:

ENVIRONMENTS: Development | Code Repository | CI Build | Test/UAT | Production

ROLES: UiPath Developer | Code Reviewer | DevOps Team | Business UAT Tester | 
Release Manager

APPROVAL GATES: Code Review | QA/Testing | Business UAT | Production Release

Output three sections:

**SECTION 1 — Diagram Layout Blueprint**
Structured blueprint with shape types, text, arrow directions, swimlanes, 
environment boundaries, and feedback loops for failed testing.

Example format:
START → Developer completes automation in DEV
↓
Decision → Code review approved?
→ YES → Build package
→ NO → Return to developer

**SECTION 2 — Mermaid Diagram**

flowchart TD
DEV[Developer Builds Automation]
COMMIT[Commit to Repo]
REVIEW{Code Review Approved?}
...

**SECTION 3 — Diagram Styling Recommendations**
- Environment grouping boxes: DEV | UAT | PROD
- Colors: Development = Blue | Testing = Yellow | Production = Green | 
  Approval Gates = Orange
- Flow direction: left → right

**SECTION 4 — Suggested Clickable Link Nodes**
Placeholder nodes for: Deployment runbook, Release checklist, UiPath packaging 
guide, Rollback process, Production approval workflow.

After this prompt I will provide the exact deployment process, approvals, 
environments, and tools used in our organization.
```

---

## Personal Productivity

### Self Review — Brutal Work Pattern Audit

> Get an honest 30-day audit of your calendar, email, and work patterns.

```text
Audit my work patterns for the last 30 days. Be brutally honest — I want the 
truth, not comfort. Analyse:

1. My calendar + emails: calculate exact hours on strategic work, meetings, 
   email, admin, and firefighting
2. Pattern analysis: what keeps showing up that I say I'll fix but don't? 
   (meetings I complain about but keep attending, commitments I make but delay)
3. Relationship Audit: who am I neglecting? (people I haven't followed up with, 
   stakeholders going cold)
4. Energy Drains: what am I doing that someone else should be doing? 
   (be specific with names/tasks)
5. The invisible work: what important stuff am I NOT doing because I'm busy 
   being busy?

Give me a performance review. Then prescribe 3 immediate changes.
```

---

## Knowledge Management

### Personal Knowledge Assistant (Second Brain)

> Set up a structured knowledge assistant that searches across all your tools and surfaces exact information.

```text
You are a personal knowledge assistant with access to the following data sources:
- Microsoft Outlook emails
- OneNote notebooks
- OneDrive/Desktop files
- Microsoft Teams messages and channels

Your role is to act as a highly organized second brain. When the user asks a 
question, you must:

RETRIEVAL BEHAVIOR:
1. Search across ALL connected sources simultaneously before responding
2. Prioritize OneNote notes as the primary source of truth
3. Cross-reference emails, Teams messages, and files to add supporting context
4. Always surface the EXACT piece of information requested — not a summary of 
   where to find it

RESPONSE FORMAT:
Structure every response using this exact template:

**Answer:**
[Direct answer to the question in 1–3 sentences]

**Supporting Evidence:**
- Source: [OneNote / Email / Teams / File]
- Date: [date of note/message/email]
- Reference: [Ticket number / Link / File name if available]
- Excerpt: [Exact relevant quote or data point from the source]

**Related Context:**
[Additional details from other sources that add useful context]

**Action Items / Follow-ups (if applicable):**
- [ ] [Any outstanding tasks, deadlines, or people to follow up with]

RULES:
- Never fabricate ticket numbers, links, dates, or names
- If information is found in multiple places, show ALL instances and note 
  discrepancies
- If nothing is found, say: "No matching records found across your connected 
  sources for: [query]" — do NOT guess
- Preserve exact ticket numbers, reference codes, URLs, and names as they 
  appear in the source
- When dates are relevant, always sort chronologically with most recent first
- If a question is vague, ask one clarifying question before searching
```

---

## Crash Course for New Topics

### Personal Knowledge Assistant (For New Topics)

> Set up a structured knowledge guide on learning new topics.

```text
I am an engineering manager who has just been handed ownership of a new initiative I have no prior context on. I need you to act as my technical onboarding coach and strategic advisor. Below I will paste all of the raw material I have been given — Jira tickets, an architectural blueprint, a requirements document, and Confluence page content. Your job is to help me get fully up to speed and ready to contribute immediately.
Once you have read everything, please give me the following:
1. Executive Summary (Plain English)
Explain what this initiative is, what problem it solves, who it affects, and where it currently stands. Write this as if I have never heard of it. Use no jargon without explaining it.
2. My Jira Tickets — Ticket-by-Ticket Breakdown
For each ticket: tell me what it is asking me to do, what "done" looks like, what decisions or dependencies I need to be aware of, and what questions I should be asking before I can make progress. Flag any tickets that seem blocked, unclear, or dependent on others.
3. What Has Already Been Done
Based on the materials, summarize the prework and decisions that have already been made so I don't re-open closed doors or duplicate effort.
4. What Still Needs to Happen
Give me a prioritized list of what needs to happen next for this initiative to move forward, and where my role fits into that.
5. Risks and Gaps I Should Know About
Based on what you see in the documents, call out anything that looks incomplete, conflicting, or risky. I want to know where the landmines are before I step on them.
6. Questions to Ask My Lead Engineer on Monday
I am meeting the engineer who was previously leading this initiative. Give me a prioritized list of questions I should ask him — focused on: filling gaps in the documentation, understanding decisions that were made and why, identifying what he is worried about, clarifying my tickets, and understanding what he needs from me to keep momentum going. Make these questions sharp and specific, not generic.
7. Meeting Prioritization Guide
I still manage my original team and cannot attend every meeting. Based on the initiative, tell me what types of meetings I absolutely must be in (decision-making, architecture, stakeholder alignment) versus which ones I can get notes from or send a delegate to.
8. My 30-Day Cheat Sheet
Give me a simple week-by-week focus guide for my first 30 days so I know where to put my energy to make the biggest impact without dropping anything.
Here is all of my source material:
[PASTE YOUR JIRA TICKETS HERE — label each one]
[PASTE YOUR ARCHITECTURAL BLUEPRINT TEXT OR DESCRIBE THE DIAGRAM HERE]
[PASTE YOUR REQUIREMENTS DOCUMENT HERE]
[PASTE YOUR CONFLUENCE PAGE CONTENT HERE — label each page]


The summary you gave me is a good starting point but I need you to go much deeper. I am going to ask you to redo several sections with far more specificity. Do not give me bullet points — write everything out in full sentences and paragraphs as if you are briefing a smart person who needs to actually act on this, not just read it.
1. Deep Dive on My Tickets — One Section Per Ticket
For each of my Jira tickets, give me a dedicated section that covers:

What this ticket is actually asking me to do in plain language, including the full scope of the work
Which specific part of the requirements document or architectural blueprint this ticket maps to — quote or reference the exact section, requirement ID, or diagram element by name so I can find it
What the logical steps are to complete this ticket from start to finish, written as a mini execution plan
What decisions I will need to make or get approved before I can close it
Who I likely need to involve or get input from based on what the documents say
What done looks like — give me a clear definition of completion so I know when to mark it closed
Any dependencies on other tickets and in what order things need to happen

2. Rebuilt 30-Day Plan — Day by Day, Not Week by Week
Rebuild the 30-day cheat sheet as a daily task list. Every single day should have between two and four specific, named tasks I can actually check off. Each task must reference a specific ticket number, a section of the requirements document, or a part of the architectural blueprint by name so
```

---
```
You are a professional Business Analyst and Onboarding Coach. Your task is to generate a polished, well-structured, and visually aesthetic Microsoft Word onboarding handbook for a new co-op joining our team. The co-op's name is Fiona.

DOCUMENT DESIGN & FORMATTING REQUIREMENTS:

Create a professional Title Page personalized for Fiona with the title: "Welcome to the Team, Fiona! — Co-op Onboarding Handbook", the team name (which I will provide below), and today's date
Use a cohesive, modern colour scheme — a deep navy or slate blue for headings with light grey/white accents
Include a Table of Contents with hyperlinked sections immediately after the title page
Use Heading 1 for major sections and Heading 2/3 for subsections
Each major application section should begin on a new page
Use callout boxes (shaded table cells or bordered text boxes) for important notes, tips, or warnings
Use clean tables for contacts, request numbers, and links — with alternating row shading for readability
Add subtle section dividers between major sections
Include a page footer on every page (except the title page) with: section name on the left, "Confidential — Internal Use Only" in the center, and page number on the right
Font: Use Calibri or Segoe UI throughout; headings bold, body text regular


DOCUMENT STRUCTURE:
Generate the following sections in order. I will fill in the bracketed placeholders with real information.

TITLE PAGE

Title: Welcome to the Team, Fiona! — Co-op Onboarding Handbook
Team Name: [INSERT TEAM NAME]
Department: [INSERT DEPARTMENT]
Manager / Buddy: [INSERT NAME]
Date: [INSERT DATE]
A brief one-line welcome message in italics underneath the title


SECTION 1 — GENERAL TEAM INFORMATION
1.1 Welcome & Introduction
Write a warm, professional 2–3 sentence welcome message for a new co-op student joining a corporate procurement/technology team. Keep it encouraging and human.
1.2 Team Contacts & Key Stakeholders
Insert a formatted table with the following columns:
| Name | Role | Email | Notes |
Fill in: [INSERT TEAM CONTACTS]
1.3 General Access Requests
These are IT/access request ticket numbers not tied to a specific application.
Insert a formatted table:
| Request # | Description | Status | Notes |
Fill in: [INSERT GENERAL REQUEST NUMBERS]
1.4 Important Links & Resources
Bullet list of general team links (SharePoint, org chart, HR portal, etc.)
Fill in: [INSERT GENERAL LINKS]
1.5 Tools & Accounts Setup Checklist
A checklist table for Fiona to track her general onboarding setup:
| Task | Done? | Notes |
Fill in as appropriate based on the contacts and requests I provide.

SECTION 2 — S2P: ARIBA
2.1 Overview
High-level description of the Source-to-Pay (S2P) process and the SAP Ariba platform — what it does, why it matters, and Fiona's exposure to it. (Write 3–5 professional sentences; I will refine as needed.)
2.2 Access Requests
| Request # | Description | Status | Notes |
Fill in: [INSERT ARIBA REQUEST NUMBERS]
2.3 Key Links & Resources
| Resource Name | URL | Notes |
Fill in: [INSERT ARIBA LINKS]

SECTION 3 — USFF CURRENT GEN (PowerApps & Power Automate)
3.1 Overview
High-level description of the USFF Current Generation solution built on Microsoft PowerApps and Power Automate — purpose, key workflows, and scope. (Write 3–5 professional sentences.)
3.2 Access Requests
| Request # | Description | Status | Notes |
Fill in: [INSERT USFF CURRENT GEN REQUEST NUMBERS]
3.3 Key Links & Resources
| Resource Name | URL | Notes |
Fill in: [INSERT USFF CURRENT GEN LINKS]

SECTION 4 — IATLK: UiPath Integration for USFF Current Gen
4.1 Overview
High-level description of the IATLK UiPath robotic process automation (RPA) integration that supports USFF Current Gen — what it automates, how it connects, and its business value. (Write 3–5 professional sentences.)
4.2 Access Requests
| Request # | Description | Status | Notes |
Fill in: [INSERT IATLK REQUEST NUMBERS]
4.3 Key Links & Resources
| Resource Name | URL | Notes |
Fill in: [INSERT IATLK LINKS]

SECTION 5 — USFF NEXT GEN (React Web App & AI)
5.1 Overview
High-level description of the USFF Next Generation platform — a modern React-based web application with AI capabilities — including its vision, current development stage, and key differentiators from Current Gen. (Write 3–5 professional sentences.)
5.2 Access Requests
| Request # | Description | Status | Notes |
Fill in: [INSERT USFF NEXT GEN REQUEST NUMBERS]
5.3 Key Links & Resources
| Resource Name | URL | Notes |
Fill in: [INSERT USFF NEXT GEN LINKS]

SECTION 6 — PROCUREMENT TRACKER (Power BI Dashboard)
6.1 Overview
High-level description of the Procurement Tracker Power BI dashboard — what metrics it tracks, who the primary users are, and how it supports procurement decision-making. (Write 3–5 professional sentences.)
6.2 Access Requests
| Request # | Description | Status | Notes |
Fill in: [INSERT PROCUREMENT TRACKER REQUEST NUMBERS]
6.3 Key Links & Resources
| Resource Name | URL | Notes |
Fill in: [INSERT PROCUREMENT TRACKER LINKS]

SECTION 7 — AZURE DATA PIPELINE (PETL)
7.1 Overview
High-level description of the Azure-based data pipeline (PETL — Procurement ETL) — its role in data ingestion, transformation, and loading, and how it supports the broader procurement ecosystem. (Write 3–5 professional sentences.)
7.2 Access Requests
| Request # | Description | Status | Notes |
Fill in: [INSERT AZURE PETL REQUEST NUMBERS]
7.3 Key Links & Resources
| Resource Name | URL | Notes |
Fill in: [INSERT AZURE PETL LINKS]

SECTION 8 — VULNERABILITIES
8.1 Overview
A professional, high-level overview of how the team manages application and infrastructure vulnerabilities — covering the general process, tools used, and Fiona's expected involvement or awareness. (Write 3–5 professional sentences.)
8.2 Active / Tracked Vulnerabilities
| Vulnerability ID | Application | Severity | Description | Status | Notes |
Fill in: [INSERT VULNERABILITY DETAILS]
8.3 Key Links & Resources
| Resource Name | URL | Notes |
Fill in: [INSERT VULNERABILITY LINKS]

CLOSING PAGE — YOU'RE ALL SET, FIONA! 🎉
Write a short, warm closing paragraph (3–4 sentences) encouraging Fiona, pointing her to her manager and team if she has questions, and wishing her a great co-op term. Keep it professional but human and upbeat.

FINAL INSTRUCTIONS:

Wherever you see [INSERT ...], leave a clearly visible placeholder in the document using highlighted text or square brackets so I can fill them in easily
Do not fabricate any request numbers, names, links, or technical details — only write the framing, overview descriptions, and structure
Apply all formatting, styles, and table designs as specified above
Make this document something Fiona would genuinely enjoy reading on her first day
```

---
*Last updated: 2026*