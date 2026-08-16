# Email to iManage vendor — open questions before build

**Version:** 1.0.0
**Date:** 15 August 2026
**Basis:** iManage Work Universal API documentation for cloudimanage.com, v2.2.26 (4 August 2026)

Everything below is a genuine gap in the published documentation, not something we
could answer by reading it more carefully. Questions 1, 4 and 5 are blockers.
Question 7 is a separate architecture discussion and can be split into its own
thread if that is easier for them.

---

## Draft email

**Subject:** iManage Work API — clarifications before we build the workspace sync

Hi [Name],

Thanks for walking us through the change-event approach. We have gone through the
Universal API documentation for cloudimanage.com (v2.2.26) and built out the calls in
Postman, and we are close to being able to start. A handful of things are not covered
in the documentation, so I wanted to get them settled before we commit to a design.

**1. How does a change-event queue relate to the cursor?**

`GET /work/api/v2/customers/{customerId}/libraries/{libraryId}/change-events` accepts a
`cursor` parameter but has no `queue_id` parameter, so creating a queue does not appear to
make the read resume from it automatically.

Our reading is that we call `GET .../change-events/queues/{queueId}` to retrieve the stored
`event_id`, pass that value as the `cursor` on the change-events request, and then write the
new cursor back with `PUT .../change-events/queues/{queueId}`. The documented example response
returns `"cursor": "2687321"`, which is the same format as an event `id`, which is what
suggests the two are interchangeable.

Can you confirm that is correct? And is the queue purely a convenience store for the position,
or does it do anything else on the server side, such as guaranteeing events are retained until
we acknowledge them?

**2. How long are change events retained?**

If our polling job is offline for an extended period, at what point does our stored cursor
become invalid and force a full reconciliation? Is retention time-based, count-based, or
configurable per tenant? This determines whether an outage costs us latency or a complete
re-crawl of the library.

**3. What are the actual rate limits on our tenant?**

We can see the `x-ratelimit-remaining` and `x-ratelimit-reset` headers documented, and we will
implement adaptive throttling against them. But we need a number to size the polling interval
before go-live rather than discovering it in production.

Specifically: what is the request ceiling and window for `GET .../change-events` and for
`GET .../workspaces/{workspaceId}`? The documentation notes limits differ per endpoint. Also,
will `rate_limiting` be `true` on the features endpoint for our environment?

**4. What type of service account will we be issued, and is `allow_logon` set?**

The documentation states the password grant requires `allow_logon` to be true on the account.
Please confirm that will be the case for the account provisioned to us.

Related: the Sync Export endpoints, including
`POST .../sync/workspaces/crawl`, are restricted to system users of type `WORK_MIGRATION`
(Library Migration) or `BACKUP_RESTORE_AGENT` (Backup and Restore). That crawl endpoint accepts
a limit of up to 1000 records, which would make our one-time initial backfill considerably
faster than paging the workspaces search endpoint.

Will our service account be one of those types? If not, we will use the standard workspaces
endpoint, which is fine, but we would rather know now than design around something we cannot use.

**5. Is the service account synchronised with an external identity provider?**

The documentation is explicit that the password grant "can only be used for users that are not
synchronized with an external IDP, or if the IDP uses the OIDC protocol with Direct Grant
support."

Please confirm either that the account sits outside your IDP sync, or that your IDP supports
OIDC Direct Grant. If neither is true, we need to discuss an alternative grant type before we go
further, since this would block the integration entirely.

Also, what token lifetime is configured for our environment? We will cache and reuse the token
rather than re-authenticating each cycle, and we would like to size that correctly.

**6. Which `custom` fields hold which business values?**

Workspace profiles expose `custom1` through `custom30`, and the mapping is configured per firm.
Could you send us the field mapping for our library, in particular which fields hold client
number, matter number, matter description, practice area and any status or closure flag? We
cannot map workspace metadata into our schema without it.

**7. Would you support us putting a message broker between the API and our consumers?**

The design you proposed is an Azure Function polling the change-event queue and writing to Azure
SQL, with the cursor persisted in SQL so the job can restart without losing its position. We are
happy with that for the current scope.

Looking further ahead, we expect to bring TyMetrix and Exterro into the same data flow. At that
point we would likely put an Azure Service Bus topic between the polling job and the downstream
consumers, so the Function's only job is to fetch from iManage and publish an event, and each
consumer subscribes independently.

Two questions on that:

- Does anything in your API terms, licensing or rate-limit policy affect us fanning out iManage
  data to multiple internal subscribers this way?
- Do you have other customers running that pattern, and is there anything you would warn us
  about, particularly around event ordering or replay?

To be clear, we are not proposing to change the current scope. We just want to make sure the
first implementation does not paint us into a corner.

---

If it is easier to cover questions 1 to 3 on a call, we are happy to do that. Questions 4, 5 and
6 we would like in writing so we have the configuration recorded.

Thanks,
Ricky

---

## Notes for us, not for the email

**Why these six are the real gaps.** We searched the full v2.2.26 documentation set. There is no
statement anywhere about event retention, no published rate-limit figures, no explanation of how
queues and cursors interact, and no webhook or push mechanism of any kind (zero results for
webhook, callback, server-sent and websocket). Polling is the only option available, so the
questions about pacing and retention are the ones that actually determine whether the design
holds.

**Question 4 is worth pushing on.** The vendor did not mention the Sync Export collection at all.
There are roughly thirty crawl endpoints there built specifically for bulk extraction, and if we
get a migration-class account the initial backfill gets meaningfully cheaper. Worth one round of
asking even if the answer is no.

**Question 7 phrasing is deliberate.** It is framed as a scope question rather than a technical
one, because the Service Bus decision is ours to make, not theirs. What we actually need from
them is confirmation that nothing in their licensing restricts redistributing the data internally.
