# Power Platform Migration Discovery — Gathering Checklist & Prompt

This document has two parts. Part 1 is what to go collect before you try to build the impact assessment. Part 2 is a ready-to-paste prompt you can run against that material once you have it (with me, or with whatever AI tool you have access to on your work network).

---

## Part 1: What to Gather

You don't need all of this before starting — even partial exports are useful. But the more of this you bring back, the less time gets spent on manual interviews later.

### A. Admin Center Exports (pull these first — they're often a few clicks and give you ground truth)

- [ ] **Power Platform Admin Center** — list of all environments (dev/test/prod), with owners
- [ ] List of all canvas + model-driven apps per environment, with owners and last-modified dates
- [ ] List of all Power Automate flows per environment, with owners, connectors used, and run status (enabled/disabled/suspended)
- [ ] List of all custom connectors and their authentication type
- [ ] Dataverse environments and table list (entity names, record counts if visible)
- [ ] DLP (data loss prevention) policies currently applied
- [ ] Connection references and who owns the underlying credentials (this is where "person left the company and now the flow is broken" risk lives)
- [ ] **SharePoint Admin Center** — list of all sites, lists, and libraries tied to these processes; site collection admins
- [ ] **Microsoft Entra ID (Azure AD)** — app registrations and service principals associated with these apps; API permissions granted
- [ ] **Exchange Admin Center** — shared mailboxes or distribution lists that workflows depend on; any mail flow/transport rules tied to these processes
- [ ] **Teams Admin Center** — Teams-integrated flows, approval apps, or bots tied to this ecosystem
- [ ] **Azure Portal** — any resource groups, Logic Apps, Function Apps, API Management instances, or Key Vaults that support these integrations

### B. People to Identify (don't interview yet — just build the list)

- [ ] Business process owner for each major app/workflow
- [ ] "Citizen developers" — people who built apps/flows informally, often without IT's knowledge
- [ ] IT/help desk contact who can pull ticket history tagged to these systems
- [ ] Procurement/vendor management contact (for any paid connectors or external APIs)
- [ ] Compliance/security contact if any of this touches regulated data

### C. Documents to Pull

- [ ] Existing process documentation or SOPs (even outdated ones are useful as a starting skeleton)
- [ ] Any prior architecture diagrams
- [ ] Excel trackers or "shadow databases" referenced in any workflow
- [ ] Evidence of email-based workflows (e.g., an inbox that triggers manual action — "send to approvals@company.com")
- [ ] SLAs or vendor contracts referencing these systems
- [ ] Any audit or compliance reports that mention these systems by name

### D. Logs and Usage Data

- [ ] Power Automate flow run history (success/failure rates) — exportable from the admin center analytics
- [ ] Any existing monitoring dashboards (Application Insights, Power BI usage reports)
- [ ] Power BI reports/dashboards that consume data from this ecosystem (helps map downstream dependencies)

### E. A Simple Tracking Table

Even a rough spreadsheet like this, filled in as you go, becomes the seed of the application inventory:

| App/Flow Name | Type (App/Flow/List/etc.) | Business Owner | Technical Owner | Department | Status (Active/Unknown/Suspected Shadow IT) |
|---|---|---|---|---|---|
| | | | | | |

---

## Part 2: The Prompt to Run

Once you have even a partial set of the above (exports, a filled-in tracking table, names of people to interview), paste it alongside this prompt into your AI tool of choice. It's designed to take messy, partial input and turn it into the structured discovery document, while flagging gaps instead of guessing.

```
Act as a senior Business Systems Analyst and Integration Specialist helping me 
build an impact assessment for migrating a Power Platform ecosystem to a React 
app on Azure.

I'm going to give you raw discovery material: admin center exports, a partial 
app/flow inventory, names of stakeholders, and notes from informal conversations. 
It will be incomplete and messy.

Your job:

1. Organize everything I give you into a structured application/integration 
   inventory: name, type, owner (business + technical), department, status, 
   and confidence level (confirmed vs. inferred vs. unknown).

2. For each item, identify what's MISSING relative to a full discovery profile 
   (business context, upstream dependencies, downstream dependencies, data 
   entities touched, security/compliance notes, known issues). Don't fill 
   gaps with assumptions — flag them as open questions I need to go ask someone.

3. Specifically hunt for indirect or hidden dependencies: flows that trigger 
   other flows, SharePoint lists referenced by multiple apps, shared mailboxes 
   acting as informal queues, Excel files standing in for a database, and any 
   app/flow with no clear owner (likely shadow IT or an orphaned process from 
   someone who's left the company).

4. Produce a running risk register as you go: anything that looks like a single 
   point of failure, an unowned process, or a dependency with no documented 
   error handling should be flagged as Critical or High risk by default until 
   proven otherwise.

5. After processing what I've given you, generate a short, prioritized list of 
   the next 5-10 questions I should ask stakeholders, ranked by how much risk 
   or ambiguity they'd resolve.

Output format: a running inventory table, a running risk register table, and a 
short list of open questions — updated each time I paste in new material.

Here's what I have so far:

[PASTE YOUR EXPORTS, TRACKING TABLE, AND NOTES HERE]
```

### How to use it

Run this once with whatever you have today, even if it's just the tracking table from Part 1 with a handful of rows filled in. The output will hand you back a prioritized question list — that becomes your interview script for the stakeholder conversations. Then come back, paste in what you learned from those interviews, and run it again. Each pass tightens the inventory and shrinks the "unknown" column.
