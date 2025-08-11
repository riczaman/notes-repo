Automation platform that lets you build, test, and deploy automations using pipelines.

## Jenkins Infrastructure 
- Master Server = Controls the pipelines & schedules builds
- Agents/Minions Servers = Run the builds 

### Jenkins Build Types
1. Freestyle Builds - shell scripts that are run on servers based on specific events.
2. Pipeline Builds - Use Jenkins files to define declaritively on how to deploy the build in different stages.

---

## Jenkins GUI
- Manage Jenkins: This contains all of the settings that you need for your Jenins instance such as plugins, global settings, etc.
- `System Configuration/Configure System`
- `System Configuration/Manage Plugins`
- `System Configuration/Manage Nodes & Cloud`: This is where you setup agents
- `Security/Manage Credentials`: This where you store `SSH keys` or `API tokens`
-  `Tools and Actions/Prepare for Shutdown`: You need to use this when you are performing and upgrade or maintanence if you just shutdown the server without doing this step then you will interrupt jobs that are running. 

---

## Setting up Freestyle Jenkins Projects
1. Go to the Jenkins Dashboard then click on  **New Item**. The two most popular types of projects are `freestyle` and `pipelines`

2. Pick the type of job and give it a name. **Make sure to not to put spaces in the name**

3. From the build options: 

      - `Source Code Management` is usually always `Git` and Jenkins will pull that repo that is specified here. You also will mention any branches if you need specific ones. 

      - Then we have `Build Triggers`: You would usually using GitHub webhooks but you need to make sure the firewall or port is open on the Jenkins server so that it can work with the webhook. `Build Periodically` is used to build jobs on a schedule using cron jobs. 

      - `Build Enviornments`: Good to select the 'Delete Workspace' option to clean up any artifacts from previous runs

4. `Build`:
      - Most common option is `Execute Shell`

5. `Post-Build`: Like email notifications

6. After clicking and saving on the build - from the job dashboard you can click `Build Now`

![Freestyle Jenkins Build GIF](/notes-repo/images/first-job.gif)

7. Click on `Configure` from the Build Homepage to change any settings

8. `Enviornment Variable`:
      - To see what env variables your build has access to go to `Configure` on the build and then scroll to Build Steps and check the `Execute Shell` step and click on "See the list of available environment variables"
      - `Main Enviornment Variables`:
            - `BUILD_ID`: Gives you the current build ID and you can use this for docker images
            - `BUILD_URL`
      - To see use enviornment variables `${VAR_NAME}` in your shell script

9. Reading Console Output:
      - Any line in the console that is prefixed with a plus sign `+` means that is a command that is being run

10. `Workspace`: If there are files that are created and managed by your build they will show up in Jenkins on the Project homepage under the Workspace folder

---

## Jenkins Filesystem
- `cd /var/jenkins_home` this will take you to the home directory of Jenkins on the **Master Server**
- Within this folder you can navigate to the `workspace` directory and this is the directory that will contain all of your builds with their job names as the folder name. Within the build folders will be any artifacts of the build.
- Within the **jenkins_home** repo also contains other important places to troubleshoot such as:
      - `plugins`: Contains a list of all the plugins installed on your Jenkins instance
      - `updates`: This contains a list of all the updates that happened
      - `logs`: Houses the log files on the server

## Setting up a Python Build with a GitHub Repo
1. Go through the same steps as above when it comes to creating a simple freestyle project but the main thing here is you want to put the GitHub URL in the `Source Code Management` section. **Note: if the repo is private then you would need to add credentials so Jenkins could clone it but if its public then no credentials are needed**

2. Since we want to execute a Python build we need to first make sure Python is installed on our master and Jenkins agents. Can do this by remoting into the server and just running the python command in the shell - `python or python3`

3. Then just run the script from the repo `python3 script_name.py`

4. **This is valuable because you can run python jobs via Jenkins anytime you want without having to SSH into servers so if you setup a scheduled build from Jenkins thats linked to a Python script from a repo you can run it fairly smooth**

---

## Setting up Jenkins Agents/Workers

1. Go to `Manage Jenkins/Nodes` and this where you will see the Jenkins Master and Nodes/Agents that are setup
**Configure Cloud** is how you build out cloud agents like Docker to use instead of pernanet ones.

2. To create a `Docker` agent go to `Cloud` then go to install plugins which will automatically filter `Cloud Providers` and install Docker then restart Jenkins

3. You can login to the `Master Jenkins Node` and go to the logs and plugins folder to also verify if the plugin is installed. Also, **refreshing the Jenkins page might need to be done as the UI will hang on the refresh portion after installing the plugin**

4. Now you can go back to the `Configure Cloud` option under nodes and add Docker.

5. Setting up Docker:
      - Need to provide the `Docker Host URI` which can be another server or if you want to do it locally you can use Docker Desktop and an alpine image in the URI field need to put in tcp://IPAddress that is generated
      **After entering in the data `set it to enabled` then `test connection`**

6. Creating the `Docker Agent Template`:
      - Go back into the configured Docker agent and then click on the gear icon and then navigate to `Docker Agent Templates` and then click on **Add**

      - Key Terms:
            - `Labels`: This is used to help the **Master** node determine which agent to send the build too.
            - `Docker Image`: This will be the official image that you will be using.
            - `Instance Capacity`: This defines how much instances of the agent will be created.
            - `Remote File System Root`: This defines where the workspace for this agent will be created - default: `/home/jenkins`

---

## Configuring Jobs to Agents

1. Go to the Job Name then click on "Configure" then under **General** click on the check box for `Restrict where this project can run` and now put in the name of the *docker agent template* then click save

2. Now when you build the job it will use the specific agent that you assigned the docker-alpine label too. **Note: It's important that you use the correct image because outdated ones will keep your job in pending state as it won't be able to find a live agent since an incorrect image will lead to provisioning errors**

3. For these Docker Jobs on the same screen of the Console Output you will see a tab called `Built on Docker` which shows the container details. 
