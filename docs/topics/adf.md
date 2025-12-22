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