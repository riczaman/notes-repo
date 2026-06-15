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

```
You are acting as an Automation & Process Optimization Analyst for my team. I want you to analyze my day-to-day work across Microsoft Teams chats/channels, Outlook emails, meeting notes, Excel files, Confluence pages, Jira tickets, SharePoint documents, Power BI workflows, deployment processes, operational tasks, and any other work artifacts you can access.

Your objective is to identify:

* Repetitive manual tasks
* Time-consuming workflows
* Processes with copy/paste work
* Tasks involving data mapping, reconciliation, validation, or reporting
* Processes involving multiple systems/tools
* Bottlenecks, delays, or dependency issues
* Opportunities for scripting, automation, dashboards, integrations, or AI-assisted workflows
* Tasks suitable for delegation to a co-op student as small-to-medium technical projects

For every opportunity you identify, provide:

1. Problem Summary
2. Current Manual Process
3. Why It’s Inefficient
4. Estimated Time Saved
5. Recommended Solution
6. Suggested Technology Stack
7. Difficulty Level (Easy / Medium / Hard)
8. Estimated Development Time
9. Whether it is suitable for a co-op/intern project
10. Potential business impact

Focus heavily on opportunities involving:

* Python scripting
* PowerShell
* APIs
* Excel automation
* Power BI automation
* Jira/Confluence integrations
* SharePoint automation
* Microsoft Graph API
* Azure/GCP automation
* DevOps/CI-CD improvements
* Reporting automation
* Vulnerability management workflows
* Infrastructure operations
* DevSecOps tasks
* Email parsing and workflow automation
* File processing
* Data cleanup and transformation
* Scheduled jobs
* Dashboard generation
* ChatOps / Teams bots
* AI-assisted summarization or classification

When reviewing my workflows, look for patterns such as:

* Weekly recurring tasks
* Manual spreadsheet updates
* Copying data between systems
* Looking up information from documentation repeatedly
* Repetitive status reporting
* Manual ticket triaging
* Manual onboarding/offboarding tasks
* Environment checks
* Compliance or audit evidence gathering
* Release coordination work
* BAU operational activities
* Manual deployment validation
* User access reviews
* Repetitive troubleshooting steps
* Manual vulnerability remediation mapping

Example of the type of opportunity I want identified:

“Every week I receive an Excel spreadsheet containing vulnerabilities. I manually review each vulnerability, search Confluence for the corresponding remediation steps, and map the resolution task back into the spreadsheet. This could be automated with a Python script that:

* Reads the Excel file
* Extracts vulnerability IDs/names
* Searches or references Confluence documentation
* Maps remediation steps automatically
* Outputs an updated spreadsheet/report”

For each identified opportunity, think beyond basic automation. Suggest:

* Intelligent workflows
* Internal tooling ideas
* Self-service dashboards
* AI copilots
* Notification systems
* Approval workflows
* Monitoring scripts
* Reporting portals
* Knowledge management improvements

At the end, provide:

* A prioritized Top 10 automation opportunities list
* Quick wins (<4 hours effort)
* Medium-impact projects
* High-impact strategic tooling ideas
* Recommended projects specifically appropriate for a co-op student
* Suggested roadmap for implementation

Be proactive and think like a senior engineer trying to eliminate operational toil and scale the team efficiently.
```
---
```
# Enterprise Business Requirements Discovery and BRD Generation Prompt

Act as a Senior Business Analyst, Enterprise Solution Architect, Product Owner, UX Analyst, and Technical Lead.

Your objective is to reverse engineer an existing Power App solution and produce a complete enterprise-grade Business Requirements Document (BRD) for integrating the Power App functionality into an existing React-based sourcing request application.

---

## Project Context

We currently have:

### Existing Application

A React-based web application used for submitting sourcing requests.

### Secondary Application

A Microsoft Power App containing business workflows, business rules, validation logic, questionnaires, decision trees, approval processes, and user experiences that partially overlap with our React application.

---

## Current Challenge

The Power App business team cannot provide complete requirements within the project timelines.

Development cannot wait for formal requirements gathering.

We must independently analyze available artifacts and create implementation-ready requirements that allow developers to begin work immediately.

The generated document should therefore:

* Reverse engineer requirements
* Infer business logic
* Document assumptions
* Identify gaps
* Generate implementation guidance
* Produce developer-ready requirements

---

## Source Material

I will provide the following artifacts:

### Artifact 1

Power App walkthrough recording

This recording contains:

* Screen navigation
* User interactions
* Conditional logic
* Data entry forms
* Validation rules
* Approval paths
* Business decisions

Analyze every screen and interaction.

Infer all business requirements visible in the recording.

---

### Artifact 2

Excel requirements document

Contains:

* Existing business requirements
* Question sets
* Data fields
* Business definitions
* Process information

Use this as supplemental information.

---

### Artifact 3

Initial BRD / Requirements PDF

This document only contains partial requirements.

Use it to validate findings but do not assume it is complete.

Identify missing functionality not represented in the PDF.

---

### Artifact 4

Application Flow Diagrams

Screenshots showing:

* React application flows
* Power App flows

Compare both applications.

Identify:

* Overlapping functionality
* Missing functionality
* New functionality required
* Opportunities for consolidation

---

# Analysis Instructions

Perform a comprehensive enterprise analysis.

Do not simply summarize the provided material.

Instead:

1. Reverse engineer functionality
2. Infer missing requirements
3. Infer business intent
4. Infer workflow logic
5. Infer user journeys
6. Infer decision trees
7. Infer validation requirements
8. Infer approval processes
9. Infer data requirements

Where information is missing:

* Make reasonable enterprise assumptions
* Clearly label assumptions
* Provide confidence levels

Example:

Assumption A-01

Confidence: High

Based on screen flow and user interactions, supplier risk assessment appears mandatory when sourcing category exceeds threshold value.

Reasoning:
[Explain reasoning]

---

# Deliverable Format

Generate a polished executive-quality BRD using TD Bank enterprise documentation standards.

Use professional formatting, headings, numbering, tables, diagrams, and visual structure.

The document should look suitable for:

* Executive review
* Architecture review
* Product review
* Developer implementation

---

# Required Sections

## Executive Summary

Include:

* Business objective
* Current state
* Future state
* Project goals
* Expected outcomes

---

## Current State Assessment

Document:

* Existing React application
* Existing Power App
* Pain points
* Duplication of functionality
* Process inefficiencies

---

## Future State Vision

Describe:

* Unified application experience
* Embedded functionality
* Streamlined workflow
* User benefits
* Business benefits

---

## Scope

### In Scope

Generate detailed list.

### Out of Scope

Generate detailed list.

---

## Stakeholders

Generate stakeholder matrix.

Include:

| Stakeholder | Role | Responsibility |
| ----------- | ---- | -------------- |

---

## Personas

Identify likely personas including:

* Requestors
* Procurement Teams
* Approvers
* Sourcing Specialists
* Administrators

---

## Functional Requirements

Generate uniquely numbered requirements.

Format:

FR-001
FR-002
FR-003

Each requirement must include:

* Description
* Business rationale
* Acceptance criteria
* Priority
* Dependencies

---

## Business Rules

Generate all discovered and inferred business rules.

Format:

BR-001
BR-002
BR-003

Include:

* Trigger
* Rule
* Outcome
* Exception handling

---

## Screen-by-Screen Analysis

For every discovered screen:

### Screen Name

Purpose

Fields

Validations

Conditional Logic

Business Rules

Actions

Error Handling

Dependencies

User Experience Notes

---

## Decision Trees

Document all conditional flows.

Use decision tree formatting.

Example:

IF Supplier Type = New
THEN Display Supplier Onboarding Section

ELSE
Skip Supplier Onboarding

---

## Data Model Requirements

Identify:

* Entities
* Relationships
* Required fields
* Optional fields
* Data ownership

Generate conceptual data model.

---

## Validation Rules

Identify:

* Required fields
* Field dependencies
* Conditional validations
* Data quality rules

---

## Workflow Requirements

Document:

* Submission workflow
* Approval workflow
* Routing workflow
* Escalation workflow

---

## Non-Functional Requirements

Include:

### Performance

### Security

### Accessibility

### Auditability

### Availability

### Scalability

### Compliance

### Logging

### Monitoring

### Data Retention

---

## React Application Integration Strategy

Recommend:

* UI integration approach
* Component strategy
* State management approach
* API requirements
* Backend requirements
* Security considerations

Provide architecture recommendations.

---

## Gap Analysis

Create a table:

| Capability | React App | Power App | Gap | Recommendation |

---

## Assumptions

Generate all inferred assumptions.

Provide:

* ID
* Description
* Confidence Level
* Reasoning

---

## Risks

Generate:

| Risk | Impact | Probability | Mitigation |

---

## Outstanding Questions

Generate a comprehensive list of questions that should eventually be validated with business stakeholders.

Group by:

* Process
* Workflow
* Security
* Data
* Reporting
* Approval Logic

---

## Developer Readiness Package

Generate:

### Epics

### Features

### User Stories

### Acceptance Criteria

### Technical Considerations

### Suggested Sprint Breakdown

---

## Architecture Recommendations

Act as a Senior Solution Architect and recommend:

* Simplification opportunities
* Consolidation opportunities
* Reusable React components
* Future-state architecture
* Scalability improvements
* Technical debt avoidance strategies

---

# Output Quality Expectations

The final document should:

* Be implementation ready
* Be developer friendly
* Be executive friendly
* Fill gaps where requirements are missing
* Clearly identify assumptions
* Clearly identify risks
* Provide enough detail for development to begin immediately

Do not provide a summary.

Generate the full BRD.
```
---
```
You are a Principal AI Prompt Engineer, Organizational Effectiveness Consultant, HR Performance Management Specialist, and Microsoft Copilot Agent Architect.

Your task is to review and redesign an employee objective-setting Copilot Agent.

I will provide:

1. Current MVP1 Instructions
2. Current MVP2 Instructions
3. Existing Suggested Prompts
4. Stakeholder Feedback

Your objective is to create a significantly improved version of the agent while preserving what works well today.

## Analysis Phase

Before generating any new instructions, perform a detailed assessment of:

* Strengths of MVP1
* Weaknesses of MVP1
* Strengths of MVP2
* Weaknesses of MVP2
* Gaps identified through stakeholder feedback
* Opportunities to improve user experience
* Opportunities to improve personalization
* Opportunities to reduce cognitive load
* Opportunities to improve role-based guidance
* Opportunities to improve objective quality

Provide a concise summary of findings before generating recommendations.

---

# Design Principles

The redesigned agent must:

### 1. Feel Human and Conversational

One of the biggest concerns is that MVP2 feels robotic and transactional.

The new version should:

* Sound like a supportive manager, coach, and objective-writing advisor.
* Use natural and professional language.
* Create a personalized experience.
* Avoid sounding scripted.
* Adapt wording based on information learned about the user.
* Explain why recommendations are being made.

The interaction should feel collaborative rather than form-driven.

---

### 2. Automatically Determine Organizational Level

The current process asks users to identify their level.

The new design should attempt to infer this automatically from available Microsoft 365 profile information whenever possible.

If the level cannot be confidently determined, the agent should ask a brief clarification question.

The objective-setting experience should dynamically adapt based on the user's role.

---

### 3. Role-Based Objective Guidance

The agent must support different objective-setting journeys based on hierarchy level.

#### Executive / CEO / Business Unit Head

Guide users through:

1. Business Priorities
2. Strategic Outcomes
3. Team Objectives (OKRs)
4. Individual Objectives

Business priorities should be established first.

Team objectives should align to those priorities.

Individual objectives should align to team objectives.

---

#### Senior Leaders

Guide users through:

1. Team Objectives
2. Department Objectives
3. Individual Objectives (if applicable)

---

#### People Managers

Guide users through:

1. Team Objectives
2. Individual Objectives

---

#### Individual Contributors

Guide users through:

1. Team Objective Alignment
2. Individual Objectives

Do not ask Individual Contributors to create business priorities.

---

### 4. Dependency Validation

Before creating objectives, verify whether the user has access to the required upstream objectives.

Examples:

* Team objectives
* Department objectives
* Business priorities

If required inputs are missing, provide guidance such as:

"Before creating your objectives, make sure you have reviewed your team's objectives or discussed priorities with your manager. This will help ensure alignment and prevent rework later."

The agent should not block progress but should strongly encourage alignment.

---

### 5. Personalization

The agent should gather enough context to make recommendations feel tailored.

Possible information to collect:

* Current role
* Team
* Key responsibilities
* Current projects
* Career aspirations
* Development interests
* Major priorities for the year

Use this information to create more meaningful and personalized objectives.

Avoid creating generic objectives.

---

### 6. Objective Quality

Objectives should be:

* Specific
* Measurable
* Action-oriented
* Realistic
* Relevant
* Time-bound

The agent should challenge vague objectives and improve them before finalizing.

---

### 7. Quarter-Based Language

Stakeholder feedback indicates the organization uses quarter-based terminology.

The redesigned agent must:

* Refer to Q1, Q2, Q3, Q4
* Refer to upcoming quarters
* Refer to quarterly checkpoints

Avoid references such as:

* H1
* H2
* Half-year reviews
* Half-year checkpoints

Unless explicitly requested by the user.

---

### 8. Suggested Prompt Optimization

Review all suggested prompts from MVP1 and MVP2.

Create a consolidated set of prompts that:

* Covers the most valuable user journeys.
* Minimizes overlap.
* Reduces the total number of prompts.
* Uses simple and intuitive language.
* Groups similar use cases together.
* Is understandable to non-technical employees.

Stakeholder feedback specifically mentions interest in:

* Understanding priorities
* Preparing for manager conversations
* Reviewing progress
* Calibration readiness
* Objective creation
* Objective refinement

Propose:

1. Final Prompt Title
2. User-Friendly Description
3. Reasoning for inclusion

Also identify prompts that should be merged or removed.

Example improvement:

Instead of:

"Review my objectives for calibration readiness"

Use language closer to:

"Check whether my team's objectives are ready for a calibration conversation"

while remaining concise enough for Copilot prompt title limits.

---

### 9. Instruction Length Constraint

Microsoft Copilot Agent Builder has instruction size limitations.

After producing the improved design:

* Generate a Full Recommended Version.
* Generate an Optimized Production Version that preserves all critical behavior while staying within platform limits.
* Explicitly estimate whether the optimized version is likely to fit within an 8,000-character instruction limit.

If content exceeds limits:

* Prioritize preserving behavior over wording.
* Compress language while maintaining intent.

---

# Deliverables

Produce the following sections:

## Section 1

Current State Assessment

## Section 2

Recommended Improvements

## Section 3

Conversation Flow Design

Include:

* Executive flow
* Leader flow
* Manager flow
* Individual contributor flow

## Section 4

Refactored Agent Instructions (Full Version)

## Section 5

Refactored Agent Instructions (Production Version Under Character Constraints)

## Section 6

Optimized Suggested Prompts

Provide:

* Final Prompt Name
* Description
* User Value
* Merged/Removed Recommendations

## Section 7

Implementation Risks and Recommendations

Identify any Microsoft Copilot Agent Builder limitations that may impact the design and recommend mitigation strategies.

Wait until all supplied MVP documents and stakeholder feedback have been reviewed before generating the redesigned solution.

```
---
```
Act as a Senior Solutions Architect, Enterprise Integration Architect, Cloud Security Architect, and Lead Technical Business Analyst.

Your objective is to create a concise interim solution requirements document (5–10 pages maximum) for a temporary integration between an existing Power Apps ecosystem and a React-based application hosted in Azure.

This interim solution must prioritize:

* Minimal development effort
* Low operational overhead
* Enterprise security compliance
* Reusability of existing application capabilities
* Ease of future migration to the strategic target-state solution

## Background

A comprehensive 40-page requirements document already exists for the long-term integration approach.

The strategic future-state solution is not currently feasible due to timeline constraints.

An interim solution is required.

### Current Environment

* Source application consists of multiple Power Apps applications and Power Automate flows.
* Power Apps persists data into SharePoint Lists.
* Target application is a React web application hosted in Azure.
* The React application has an ASP.NET backend and its own database.
* The target environment is network-restricted and does not allow inbound connectivity from Power Apps.
* Because of network restrictions, the source application cannot push data directly into the target application.

### Interim Business Requirement

Every hour:

1. Retrieve new user records from the source system.
2. Create a draft intake within the target application.
3. Store the intake in the target database.
4. Send a notification email to the user informing them that a draft intake has been created.

### Candidate Solutions

#### Option A – SharePoint Pull Model

* An Azure-hosted scheduled process reads SharePoint List data hourly.
* The scheduled process uses Microsoft Graph or SharePoint APIs.
* The process calls an internal intake API exposed by the target application.
* The target application performs validation and persistence.
* The target application sends notification emails.

#### Option B – Email File Transfer Model

* The source application sends hourly emails with attached files.
* A mailbox is monitored by a scheduled process.
* The scheduled process parses attachments.
* Draft intakes are created in the target application.
* Notification emails are sent to end users.

## Existing Information

[INSERT SUMMARY FROM EXISTING REQUIREMENTS DOCUMENT]

Include:

* Intake data model
* Required fields
* Volume estimates
* Current business workflow
* Email requirements
* Network constraints
* Existing APIs
* Security requirements
* Compliance requirements
* Retention requirements
* SharePoint schema

## Required Deliverables

Produce the following sections.

### 1. Executive Summary

Provide a concise summary of the interim solution.

### 2. Problem Statement

Describe the business problem and constraints.

### 3. Assumptions

Document all technical and business assumptions.

### 4. In Scope

Define the scope of the interim solution.

### 5. Out of Scope

Clearly identify future-state capabilities that are excluded.

### 6. Business Requirements

Document functional and non-functional requirements.

Include:

* Batch frequency
* Intake creation requirements
* Email notification requirements
* Error handling expectations
* Audit requirements
* Reporting requirements

### 7. Architecture Decision Analysis

Compare Option A and Option B.

Evaluate:

* Security
* Operational complexity
* Cost
* Reliability
* Scalability
* Maintainability
* Compliance
* Future migration effort

Provide a weighted decision matrix.

Select the preferred option and justify the recommendation.

### 8. High-Level Solution Design

Include:

* End-to-end workflow
* Sequence of events
* Major components
* Integration points
* Authentication approach
* Error handling approach
* Monitoring approach

### 9. Security and Identity Requirements

Identify:

* Required Microsoft Graph permissions
* SharePoint permissions
* Entra ID requirements
* Managed identity requirements
* Service account requirements
* Network connectivity requirements

Recommend least-privilege access patterns.

### 10. API Strategy

Determine whether:

* Microsoft Graph APIs are sufficient
* SharePoint REST APIs are required
* A custom API should be created
* Existing internal APIs can be reused

Provide recommendations.

### 11. Risks and Mitigations

Document:

* Technical risks
* Operational risks
* Security risks
* Data quality risks
* Interim solution permanence risks

Provide mitigations.

### 12. Team Dependencies

Identify all required stakeholder teams and responsibilities.

### 13. Open Questions

Generate a list of unanswered questions required before implementation can begin.

### 14. Developer Implementation Guidance

Provide high-level implementation guidance only.

Include:

* Recommended Azure services
* Batch orchestration pattern
* Idempotency strategy
* Retry strategy
* Monitoring approach
* Logging approach
* Deployment considerations

Do not provide low-level code.

## Output Requirements

* Use concise language.
* Focus on decision-making information.
* Challenge assumptions where appropriate.
* Identify hidden dependencies.
* Highlight security implications.
* Explicitly call out areas requiring validation.
* Ensure the interim solution can be decommissioned easily when the strategic solution is implemented.

```

*Last updated: 2026*