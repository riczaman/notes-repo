# iManage Integration Architecture Options

## Objective

Build middleware that detects changes in **iManage**, retrieves matter data, transforms it as required, and sends it to **Exterro** and **TyMetrix**.

The integration should also be extensible so additional downstream applications can be added later.

---

# 1. Option Summary

There are two viable Azure approaches:

### Option 1: Azure Data Factory

Use ADF as the orchestration layer:

```text
iManage
   │
   │ Scheduled polling
   ▼
Azure Data Factory
   │
   ├── Retrieve data
   ├── Transform
   ├── Call Exterro
   └── Call TyMetrix
```

### Option 2: Azure Functions + Service Bus

Use application code for integration logic and Service Bus for asynchronous distribution:

```text
                    iManage
                       │
                 Poll for changes
                       │
                       ▼
                Azure Function
                       │
                       ▼
              Canonical Matter
                       │
                       ▼
              Azure Service Bus
                   Topic
                  /     \
                 /       \
                ▼         ▼
           Exterro      TyMetrix
           Function     Function
```

**Both approaches are technically feasible.**

The key difference is that **ADF is faster to initially build**, while **Functions + Service Bus provides a stronger long-term architecture for an application integration platform.**

---

# 2. Option 1: Azure Data Factory

## Architecture

```text
iManage
   │
   ▼
ADF Scheduled Pipeline
   │
   ├── Poll iManage
   │
   ├── Retrieve Matter
   │
   ├── Transform Data
   │
   ├── Call Exterro
   │
   └── Call TyMetrix
```

## Pros

- Faster initial development
- Visual/low-code pipeline development
- Built-in scheduling
- Built-in API/data activities
- Straightforward for relatively simple integrations
- Familiar Azure service
- Less custom application code required initially

## Cons

- Application logic can become embedded in pipelines
- Retry and error handling becomes pipeline-centric
- Downstream systems are more tightly coupled to the orchestration
- More difficult to independently process Exterro and TyMetrix
- Pipeline complexity grows as additional applications are added
- Less natural fit for complex application business logic
- Automated unit testing is less natural than with application code
- ADF becomes increasingly difficult to manage if it evolves into a general-purpose integration hub

## Best Fit

ADF is a reasonable choice if:

- The integration is relatively simple
- There will only be a small number of downstream systems
- Transformations are straightforward
- Polling frequency is relatively low
- Future expansion is limited
- Speed to the first working version is the primary concern

---

# 3. Option 2: Azure Functions + Service Bus

## Architecture

```text
                         iManage
                            │
                    Poll for changes
                            │
                            ▼
                    Azure Function
                            │
                            ▼
                  Canonical Matter Model
                            │
                            ▼
                   Azure Service Bus
                        Topic
                      /       \
                     /         \
                    ▼           ▼
             Exterro Function  TyMetrix Function
                    │           │
                    ▼           ▼
                 Exterro      TyMetrix
```

## Pros

- Strong separation between iManage and downstream systems
- Independent processing for Exterro and TyMetrix
- Durable asynchronous messaging
- Better retry and failure isolation
- Dead-letter handling
- Easier replay/reprocessing
- Better fit for complex business logic
- Standard application code can be unit tested
- Easier to add future downstream applications
- Better long-term maintainability as integrations grow

## Cons

- More initial developer effort
- Requires application development rather than primarily visual configuration
- Developers must implement polling and state management
- Developers must implement transformation logic
- More Azure components to configure and operate
- Requires appropriate CI/CD and application deployment practices

## Best Fit

This approach is better when:

- The integration is expected to grow
- Additional downstream applications are likely
- Different applications require different transformations
- Independent retry/failure handling is important
- Integration reliability is important
- There is meaningful business logic
- The organization wants an extensible integration platform

---

# 4. Why ADF Is Not the Preferred Long-Term Middleware

ADF is **not a bad technology for this problem**.

The issue is that the iManage requirement is fundamentally an **application integration problem**, not simply a data pipeline.

The initial ADF implementation could be very efficient:

```text
Poll → Transform → Exterro → TyMetrix
```

However, over time the pipeline could become responsible for:

```text
Polling
Transformation
Business rules
API orchestration
Retries
Error handling
Routing
Application-specific mappings
Downstream dependencies
Integration state
```

This makes ADF increasingly act like a custom application integration engine.

The Functions + Service Bus architecture separates these responsibilities:

```text
iManage
   │
   ▼
Change Detection
   │
   ▼
Canonical Matter
   │
   ▼
Service Bus
   │
   ├── Exterro Adapter
   │
   ├── TyMetrix Adapter
   │
   └── Future Adapter
```

This means adding a new downstream application does not require redesigning the core iManage integration.

---

# 5. Developer Work Required

## Option 1: ADF

### Main development tasks

1. Create ADF environment and pipeline.
2. Configure scheduled polling of iManage.
3. Implement iManage authentication.
4. Implement pagination and change detection.
5. Implement watermark/state handling.
6. Transform iManage data.
7. Build Exterro API integration.
8. Build TyMetrix API integration.
9. Configure retries and error handling.
10. Add logging and monitoring.
11. Implement manual sync capability if required.

### Development profile

**Lower initial coding effort.**

More work is done through ADF configuration and pipeline activities.

---

# 6. Developer Work Required: Functions + Service Bus

### Main development tasks

1. Create Azure Function App.
2. Build iManage polling Function.
3. Implement iManage authentication.
4. Implement pagination and change detection.
5. Implement durable polling watermark/state.
6. Create canonical Matter model.
7. Implement iManage → canonical transformation.
8. Create Service Bus topic.
9. Create Exterro subscription.
10. Create TyMetrix subscription.
11. Build Exterro adapter Function.
12. Build TyMetrix adapter Function.
13. Implement retries and error handling.
14. Implement idempotency/duplicate handling.
15. Implement dead-letter handling.
16. Create integration status/state storage.
17. Add API Management endpoint for manual sync.
18. Add logging, monitoring and correlation IDs.
19. Add automated unit/integration tests.
20. Configure CI/CD deployment.

### Development profile

**Higher initial coding effort**, but the resulting components are more modular and easier to extend.

---

# 7. High-Level Development Comparison

| Area | ADF | Functions + Service Bus |
|---|---|---|
| Initial setup | Faster | Moderate |
| Coding required | Lower | Higher |
| Polling | Configuration + some logic | Custom Function |
| Transformations | ADF mapping | Application code |
| Exterro integration | Pipeline/API activity | Dedicated adapter |
| TyMetrix integration | Pipeline/API activity | Dedicated adapter |
| Retry handling | Pipeline configuration | Messaging + application logic |
| Failure isolation | Moderate | Strong |
| Dead-letter/replay | More custom | Natural fit |
| Testing | Pipeline-oriented | Standard application testing |
| Future applications | Increasing pipeline complexity | Add subscription + adapter |
| Long-term extensibility | Moderate | Strong |
| Operational integration fit | Good | Excellent |

---

# 8. Recommended Solution

## Recommendation

**Azure Functions + Azure Service Bus**

Supporting services:

- Azure API Management
- Azure SQL
- Azure Key Vault
- Application Insights / Azure Monitor

## Recommended architecture

```text
                         iManage
                            │
                 ┌──────────┴──────────┐
                 │                     │
             Automatic             Manual Sync
              Polling                from UI
                 │                     │
                 ▼                     ▼
          Azure Function        API Management
                 │                     │
                 └──────────┬──────────┘
                            ▼
                     Azure Functions
                            │
                            ▼
                   Canonical Matter
                       Model
                            │
                            ▼
                   Azure Service Bus
                         Topic
                      /         \
                     ▼           ▼
                Exterro       TyMetrix
                Adapter       Adapter
                     │           │
                     ▼           ▼
                  Exterro      TyMetrix
```

---

# 9. Why This Is the Recommended Architecture

The primary reason is **decoupling**.

The iManage integration should not need to know how every downstream application works.

Instead:

```text
iManage
   ↓
Canonical Matter
   ↓
Message
   ↓
Downstream adapters
```

Each adapter owns its application-specific:

- API calls
- Schema mapping
- Validation
- Retry behavior
- Error handling

If a third application is introduced:

```text
                    Service Bus
                  /      |       \
                 /       |        \
                ▼        ▼         ▼
            Exterro   TyMetrix   New App
```

The core iManage integration does not need to be redesigned.

---

# 10. Important Architectural Trade-Off

The decision is ultimately:

### ADF

**Optimize for speed and simplicity now.**

vs.

### Functions + Service Bus

**Optimize for flexibility, reliability and extensibility over time.**

ADF may produce the first working integration faster.

Functions + Service Bus requires more developer work up front, but creates a more appropriate foundation if the strategic goal is to build an **iManage integration layer that will support multiple downstream SaaS applications.**

---

# 11. Final Recommendation

> **Use Azure Data Factory if the goal is the fastest and simplest implementation of a small, stable integration.**

> **Use Azure Functions + Service Bus if the goal is to establish a reusable integration layer between iManage and multiple downstream SaaS applications.**

For the stated requirements, **Azure Functions + Service Bus is the recommended solution** because the architecture is better aligned to API-based application integration, independent downstream processing, failure isolation, extensibility, and long-term maintainability.