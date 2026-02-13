```
Copy Activity
Need your data set
A connection from your source to the copy activity 
A connection from your copy activity to the destination 
Go to the Author Tab
Data flows are used to transform data
Power Query can also be used to transform data
The term Sink = Destination in Azure Data Factory
To see your storage make sure to go to the storage account and then click on data storage > containers
Click on Pipelines
Click on new pipeline
Search for the copy activity
If your data is in the storage account then your source snd sink should be azure data lake storage 2 and then select the type of data so in this case its csv
On this screen you can choose the directory of where the file resides and if it doesn’t already exist you can create folders by supplying a name
You also need to have a previously setup linked service so that you can supply the linked service data location in this source and sink fields
You usually select none for the schema type
Running in debug mode will run the full activity which will copy the file from your source to dest. You monitor the pipeline status to see if its working 

Copying Data from an API/HTTP Connection to Data Lake (Storage Account)
First thing you have to do is create a linked service to get data from your api/http activity
So when you create a linked service choose HTTP (rest api is another option)
If its a csv on GitHub make sure you view it as Raw and then provide this URL (only up to .com) as the base URL
The rest of the URL is the data set
Since GitHub repo is public you can set the authentication type as anonymous 
Then go create the new pipeline it will only be the copy activity
In the source you would click new and then HTTP and then provide the GitHub linked service
Then you can provide the relative URL (rest of the URL after the base) and then one good thing to do is preview data to see if it can actually fetch your data
Then setup the sink using the previous linked service to your data lake
You can edit settings by clicking on the open option beside the parameter. Its important to note that you should specify the file name or it will copy the whole GitHub repo hierarchy

Publishing All
This saves all your work 

Get Metadata Activity
You can sort files into different containers using this activity 
Use case would be to create a dataset that contains all files in the directory that you want to sort
You also have to supply the field list
Child items will cover all files within a folder
After you run the activity you can see the pipeline in the bottom under the successful status and just click on the see output button
The output of this activity is an array

IF condition activity 
Allows you to run if conditions 

For Each Activity
Allows your to loop through data 

Sorting files in a container and moving them to a different one if they don’t match
Each activity in ADF has 4 options (Connections - on error, on success, on completion)
This is how you connect activities to each other
Start with a get metadata activity to get an array of files with properties
Then you need a for loop to loop through each file in the array 
To get the output variable you have two options you can use the pipeline expression builder to build out code or in the activity you select the add dynamic content and this has a list of a built in outputs that you can retrieve into a variable. So in our case it would be the get output metadata but you need to make sure you specify that you just want the child items because you don’t want the whole object just the child items array
To perform activities within the foreach you click on the for each go to the activities tab then click on the pencil button 
Anytime you want to use output from another activity you just need to add the dynamic content 
Pipeline expression builder also contains a bunch of functions you can leverage like string functions
Within the for each you need to start with an if activity
The expression for this should be dynamic content (each for item).startswith()

Parameterized DataSets
Basically in our copy activity for each file that meets the condition it needs to be copied but in our current setup we hardcode the file name which won’t work since file names will differ so we need to capture the filename in a dynamic variable = parameter
To do this: When setting your properties click on Advanced and then click on open dataset
From here you can see the tab called parameters where you can create the parameter 
Then in the file path add dynamic content and select the parameter you created
When referring to dynamic variables - you need the @ prefix
If you get a bad request when trying to run the pipeline that means you have an error somewhere in your code even if your validation passes.

Dataflow Activities:
Uses spark to transform data within your pipeline
Go to settings then setup the source first
To use spark you need to enable the data flow debug option on
After you enable the cluster but turning data flow debug on then you need to go to the projection tab and import the projection which is the schema
The data preview shows you all of the fields in your table and their type
You click on the small plus button beside the source activity of the dataflow activity then you can select what type of transformation you want to do. Transformation is basically like doing a query on the data.
SELECT you can check box any fields you want to remove
FILTER use the expression query to filter out data you don’t want
Conditional Split - splits data based on the condition
Replacing NULL: Derived Column and then select the column and in the expression you can use the coalesce function which filters out NULL and puts in the second string argument you provide 
Group By = Aggregate
Writing the Data: Select the sink destination as the last activity in the flow
Before writing to sink you should always use the ALTER Rows activity first. 
Alter condition = insert if will only insert the data based on the condition 
Use case: take your data select it all but remove 2 misc columns then filter out cust id 2 then conditionally split on payment and in amex get rid of nulls then group the main split on customer id and find the highest product id

Trigger Activities 
On the main pipeline screen just click on “Add Trigger”
When setting up the trigger select an end date
To see if the trigger actually got kicked off go to the left hand menu and click on the Managed Tab then click on triggers
Under the monitor tab you can see the pipeline runs

Set Variable Activity 
Used to store variables that can be used somewhere else in the pipeline downstream
You can set variables in the pipeline by clicking on the blank space and then going to the variables tab and then create them here
Then you use the set variable activity to set the value
Pipeline return value types are variables that can be used in other pipelines

Storage Events Trigger
A trigger based on an even as opposed to in a timely fashion 
Storage accounts specifically get triggered by changes in the data storage repo you provide 
If the file stays in the same initial repo the trigger will keep firing off so you need to delete the file after the trigger runs 
DELETE Activity
To use this storage event trigger you need to register this by managing your subscription
Subscriptions > resource providers > Microsoft.eventgrid 

Execute Pipeline
This activity lets you execute multiple pipelines together

```
---
```
ADF with GitHub Actions as the CD
Within ADF you can setup your git connection from either the live mode option at the top or from the management tab
Just need to provide the base GitHub repo name and then it will connect and you select what branch you want to start off with. 
You can create a new branch from within ADF 
Also allows you to create PRs from ADF
In Azure Portal
Go to managed identities 
Create a new one
In the resource group go to the IAM (access control) and add a role assignment and go to job function roles and search for data factory contributor.
In role assignments you can see the different roles in the resource group
On the new identity you created click into it 
Then go to federated credentials
Within here select the configure a GitHub issued token to impersonate this app and deploy to azure
Creating the CI/CD 
In the GitHub repo
Create a folder called 
Build
Then within here create a package.json
Here is the script copied from the Microsoft documentation specifically for ADF
Within the GitHub repo go to settings to setup the credentials to connect to Azure
Go to security and then secrets and variables
Then go to Actions
Create the new repository secrets 
You need to add the azure subscription id to connect the deployments with action and your azure tenant
The Azure subscription ID is provided from the managed ID
Azure client ID is also within the managed ID
Azure Tenant ID is found by searching for the entra ID within azure portal
Go to GitHub Actions
Setup the workflow - if you do it through the UI it will create the empty yml file for you if not then you will need to create a repo called workflows within the .github folder (don’t forget the period) and then create a main.yml file
Then create the workflow and embed the secrets that you need - you can get this from Microsoft documentation 
One of the first steps need to be to create the ARM template as that is what we want to use to do the deployments using actions to azure. 
Then go to GitHub actions and run the build for the first time
The deployment will deploy onto another ADF resource usually the one you create for PAT/PROD

Azure App Services
Fully managed service to deploy web apps and APIs without the need to manage infrastructure
PaaS
Your web app is stored in a index file and folder and this is served to the end user by the app service plan
App service plan vs app service
The plan provides the infra that the app service needs. The app service is your actual application
In Azure Portal
Create app service plan first and here you will use the resource group you have for app services
Now go to app services within azure portal and create a new app service
There’s. Web app, static web app, web app and data base ,and Wordpress
After you input the base info you can go ahead and create it 
Once the app service resource is created click on it
Then you can see the default domain as well as other high level info
Within the rescue search advanced tools and then click on go
It launches the KUDU environment and here is where you can use the console to add your code
Go to debug console and go to CMD and then from within here go to home\site and go to the wwwroot folder and then copy all of your code files into thes repo 
Go back to overview and check the default domain 

Kafka
When your micro service architecture can’t handle load is where we introduce the need for Kafka
Type coupling occurs when one service depends on other services so if it one goes down it causes the other services to fail
Kafka sits in the middle of the services and collects events from services and then makes these events available to the other services that need them
Events are key value pairs and have metadata in them
Producers create the events
When producers create events they give them to Kafka and these events get saved in topics which group the same type of events together
You create the topics and decide how to group the events together
Microservices subscribe to topics so Kafka will notify subscribed micro services to updates that happen in topics. 
When a service updates into the database and produces an event and this event and status is captured by Kafka
One event creates a chain of events
Kafka also allows for real time analytics 
Kafka uses stream APIs for real time analytics
Kafka has partition capabilities that allow for scaling possible especially for large amounts of data
Partitions add more workers per topic to help process 
Producers can write into partitions and consumer groups can all consume from Kafka partitions 
Data in topics are saved on Kafka servers called brokers 
Regular message queues delete messages after message consumption but in Kafka you can store messages for as long as you need for later analysis 
Kafka uses zookeeper 
But now they use Kraft for centralized control and coordination 

TIBCO
Used for file transfers
Platform server allows you to transfer files between different server environments 
Also has event driven interactions
Internet Server allows for file transfers in and out of the organization 
Command Center is the centralized control over everything that is happening in regards to file transfers
```
---

```
I have a set of raw notes exported from Notepad++. The notes contain request numbers, action items, descriptions, dates, statuses, owners, and other related details.

Please analyze the entire dataset and transform it into a properly structured Excel workbook with the following requirements:

1. Main Sheet – “Request Tracker”

Extract and organize all identifiable requests into rows.

Each request should have its own row.

Create columns for:

Request Number

Title / Summary

Description

Action Items

Owner (if present)

Status (if present)

Priority (if present)

Created Date (if present)

Due Date (if present)

Other Relevant Dates

Notes / Additional Context

Ensure all request numbers are captured and no duplicates are created unless clearly separate entries.

If multiple action items exist for a single request, separate them clearly within the same cell using bullet formatting.

2. Formatting & Organization

Convert the dataset into an Excel Table.

Apply color coding:

Red for Overdue items (based on Due Date)

Yellow for Due within 7 days

Green for Completed items

Blue for In Progress

Freeze the header row.

Enable filters on all columns.

Auto-adjust column widths for readability.

Sort by Due Date (earliest first).

3. Summary Sheet – “Dashboard”
Create a second sheet that includes:

Total number of requests

Number of Open / In Progress / Completed

Number of Overdue items

Upcoming items due in next 7 days

A simple pivot summary by Status

4. Unstructured / Ungrouped Data Sheet

If there is any data that cannot confidently be grouped into a specific request,

Place it into a separate sheet named “Uncategorized Data”

Include a column explaining why it could not be grouped (e.g., missing request number, unclear association, duplicate ambiguity, etc.)

5. Data Validation & Integrity

Flag any rows missing a Request Number.

Highlight missing Due Dates where action items exist.

Identify possible duplicate request numbers in a separate section.

Ensure the final Excel file is clean, professional, easy to maintain, and suitable for tracking operational or project work.

If You Want an Even More Advanced Version

If your notes are messy and inconsistent, use this upgraded version:

The notes may contain inconsistent formatting, shorthand references, and mixed paragraph structures.
Please intelligently infer relationships between request numbers, dates, and action items.
If multiple entries refer to the same request number, merge them logically.
Do not lose any data during transformation.
Prioritize accuracy over assumptions and clearly flag any inferred associations.```
----
Copilot Prompt – Outlook + Teams Action Intelligence

Analyze all my Outlook emails and Microsoft Teams chats from the past 14 days.

I only want communications that:

Were sent directly to me, mentioned me, or assigned me work

Require an action, follow-up, response, or decision from me

Please perform the following:

1. Identify and Extract Action Items

Extract clear action items assigned to me.

Infer action items where the expectation is implied (e.g., “Can you review…”, “Let me know…”, “We need your approval…”).

Include:

Action Description

Source (Email or Teams)

Sender

Date

Due Date (if mentioned or inferred)

Related Project / Topic

2. Intelligently Group by Topic

Group conversations that relate to the same subject into a single cohesive block.

For example, if there are emails and Teams chats about “PETL pipeline,” group them together regardless of platform.

Merge overlapping discussions into one comprehensive summary.

Ensure no duplicated action items appear across groups.

3. Provide Structured Output
Organize the response into:

Topic / Project Name

Summary of what happened in the last 2 weeks

Key decisions made

Current status

All action items assigned to me

Outstanding risks or blockers

Relevant links to emails or Teams threads

4. Prioritization

Identify overdue items

Highlight urgent or time-sensitive actions

Call out items with unclear expectations or missing due dates

5. Executive Summary
At the top, provide:

Total number of active topics

Total number of action items

Number of high-priority or overdue items

Top 3 themes consuming most discussion time

6. Deduplication and Intelligence

Do not list the same action twice if it appears in both email and Teams.

If discussions evolved, provide the most up-to-date context.

Flag ambiguous or unclear tasks separately under “Needs Clarification.”

The final output should read like a structured operational brief that helps me quickly understand everything I am accountable for and what requires attention next.

More Advanced Version (If You Want Strategic Insight)

Add this at the end:

Additionally, identify patterns in the last 2 weeks:

Recurring bottlenecks

Stakeholders who frequently require follow-ups

Topics that may need a formal meeting instead of fragmented chat/email discussion

Suggest how I can improve communication flow or reduce reactive workload.