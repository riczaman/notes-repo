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

4. This helps you troubleshoot because some agents might not have the required software like Python3 so if you assign this agent template that does not have python to a build that requires it then it will error out. **In this scenario you should create your own Docker Image that has python installed**

5. To create another agent you just need to go back to the `Docker Agent Template` and then create another template.

![Cloud Agent Templates](/notes-repo/images/cloud-agent.gif)
--- 

## Adding Jenkins Triggers

1. Build Triggers/`Poll SCM`: Jenkins Master will periodically check github for any changes and its much easier to manage then setting up webhooks.
    - It uses cron notation 

---

## Setting up Jenkin Pipelines

1. Similair to how we created freestyle projects you need to go to `New Item` and then click on `Pipeline`. You will notice that at the top most of the settings are the same but you have less freedom with pipelines for advanced settings as most of the steps are carried out by the **`Pipeline Script`** section.

2. There are two ways to build out the pipeline script and both ways use the **`Groovy Syntax`**:
      - `Directly in the UI`
      - `Jenkinsfile`

---

## Jenkin Pipeline Syntax

- Everything is wrapped in a `pipeline` parameter {}
- First step in the pipeline is to select the `agent` that will carry out the job and it is specified by the `label` parameter
- Next step is the `Stages`: This is where you define your stages like building -> testing -> deploying
- When you create a Pipeline Build you can see a section for **Pipleline Overview** which will show you all of the stages you built. 

---

## Jenkinsfile

- Instead of putting the script directly into the UI of Jenkins you can create a `Jenkinsfile` within the parent folder of your code repository and then outline all of the deployment steps in this special file

- Within the Pipeline build in Jenkins, you need to change the pipeline from using *Pipeline Script* to use `Pipeline Script from SCM` then add the github repo. **Make sure you put the `Jenkinsfile` path in the `Script Path`**

- When you get the pipeline from SCM it will add a first step where it sees if it can checkout from SCM without any errors. 

```groovy title="Standardized Jenkinsfile" linenums="1"
pipeline {
    agent any  // Use any available Jenkins agent/node

    // Parameters allow customization without changing the Jenkinsfile
    parameters {
        string(name: 'GIT_REPO', defaultValue: 'https://github.com/your-org/your-repo.git', description: 'GitHub repository URL')
        string(name: 'GIT_BRANCH', defaultValue: 'develop', description: 'Git branch to build')
        string(name: 'GITHUB_CREDENTIALS_ID', defaultValue: 'github-credentials-id', description: 'Jenkins credentials ID for GitHub access')
        string(name: 'SONARQUBE_PROJECT_KEY', defaultValue: 'my-app', description: 'SonarQube project key')
        string(name: 'SONARQUBE_TOKEN', defaultValue: 'sonarqube-token-id', description: 'Jenkins credentials ID for SonarQube token')
        choice(name: 'BUILD_ENV', choices: ['dev', 'qa', 'prod'], description: 'Deployment environment')
    }

    environment {
        // Common environment variables for builds
        NODE_VERSION = '18'
        JAVA_VERSION = '17'
        MAVEN_HOME = tool name: 'Maven 3', type: 'maven'
        // This assumes Jenkins SonarQube plugin is configured as "sonarqube"
        SONARQUBE_ENV = 'sonarqube'
    }

    triggers {
        // Automatically build when there are new commits in the repo
        pollSCM('H/5 * * * *') // Every 5 minutes
    }

    stages {
        stage('Checkout') {
            steps {
                // Clone code from GitHub using stored credentials
                git branch: "${params.GIT_BRANCH}",
                    credentialsId: "${params.GITHUB_CREDENTIALS_ID}",
                    url: "${params.GIT_REPO}"
            }
        }

        stage('Set Up Node') {
            steps {
                // Install Node version for frontend builds
                sh """
                    echo "Setting up Node.js ${NODE_VERSION}"
                    nvm install ${NODE_VERSION}
                    nvm use ${NODE_VERSION}
                """
            }
        }

        stage('Set Up Java') {
            steps {
                // Configure Java for backend builds
                sh "java -version"
                sh "javac -version"
            }
        }

        stage('Install Frontend Dependencies') {
            when { expression { fileExists('frontend/package.json') } }
            steps {
                dir('frontend') {
                    sh "npm install"
                }
            }
        }

        stage('Build Backend') {
            when { expression { fileExists('backend/pom.xml') } }
            steps {
                dir('backend') {
                    sh "${MAVEN_HOME}/bin/mvn clean install -DskipTests"
                }
            }
        }

        stage('Run Unit Tests') {
            parallel {
                stage('Frontend Tests') {
                    when { expression { fileExists('frontend/package.json') } }
                    steps {
                        dir('frontend') {
                            sh "npm test"
                        }
                    }
                }
                stage('Backend Tests') {
                    when { expression { fileExists('backend/pom.xml') } }
                    steps {
                        dir('backend') {
                            sh "${MAVEN_HOME}/bin/mvn test"
                        }
                    }
                }
            }
        }

        stage('SonarQube Analysis') {
            steps {
                withSonarQubeEnv("${SONARQUBE_ENV}") {
                    withCredentials([string(credentialsId: "${params.SONARQUBE_TOKEN}", variable: 'SONAR_TOKEN')]) {
                        dir('backend') {
                            // You can run separate Sonar scans for frontend/backend if needed
                            sh """
                                ${MAVEN_HOME}/bin/mvn sonar:sonar \
                                    -Dsonar.projectKey=${params.SONARQUBE_PROJECT_KEY} \
                                    -Dsonar.login=$SONAR_TOKEN
                            """
                        }
                    }
                }
            }
        }

        stage('Conditional Deployment') {
            when {
                anyOf {
                    branch pattern: "feature/.*", comparator: "REGEXP"
                    branch pattern: "hotfix/.*", comparator: "REGEXP"
                    branch 'main'
                    branch 'develop'
                }
            }
            steps {
                sh """
                    echo "Deploying to ${params.BUILD_ENV} environment"
                    # Add deployment scripts or commands here
                """
            }
        }
    }

    post {
        success {
            echo "Pipeline completed successfully."
        }
        failure {
            echo "Pipeline failed. Check logs for details."
        }
        always {
            cleanWs() // Clean workspace after build
        }
    }
}
```

---

## Settings up Jenkins Agents on Windows Server

- Need to make sure the version of Java that you install on the Windows Server is the same one that is installed on the Jenkins Controller
- To find the version, go to `Manage Jenkins` then go to `System Information` and search for **java.home**
- Adding these Nodes:
      1. Click on Manage Configurations and then go to Nodes and create a new node
      2. Provide a name and then click on `Permanent` this it is not a cloud or dynamic server.
      3. The remote root directory will be a path on the server: `d:\tools\jenkins-agent`
      4. If you leave the `label` blank it will take on the name of the agent.
      5. Usage needs to be set as only when using the label name
      6. `Websocket` If you don't select this option then you need to just make sure you open a specific port on the agent so that it can communicate with the master via `tcp`. When you save this in the Jenkins Master it will save the connection type but will be marked with a red X because you need to connect the agent to the master. **If you click on the red X it will give you commands to run on the agent**

      7. For exectuable services you can leverage `WinSW` which is a wrapper for any executable so that it can be run as a Windows Service
         - The way this works is you rename the WinSW ex as your jenkins agent and then need an xml file that defines what it runs
         - The arguments in the xml come from the Jenkins UI when you click on the red X
         - The agent.jar also comes from this locaton after you click on the red X
         - After you start the service then it will connect to the master

---

## Jenkins Multibranch Pipelines

1. Creating a `GitHub App`: Go to GitHub and click on Settings then go to Developer Settings then create the GitHub App and Set the permissions. Then set the subscribe events like push, repository, and other events.

2. To add credentials in Jenkins: Click on `Manage Jenkins\Credentials` then click on the Global one and then add credential and add for a GitHub app.
**PAT (personal access token) gives you a lot less GitHub limits whereas a GitHub app lets you call to GitHub a lot more**

3. Create a new item and select `Multi-Branch Pipeline`:
      - For branch sources select GitHub
      - *If you dont have a Jenkinsfile then there will be no build configuration since you are selecting builds based on this type of file*

4. The way the multibranch pipeline works is that Jenkins scans the repo for every branch name and allows you to see that in the app homepage view:
      - `On Different Branches you can modify the Jenkinsfile so that different things can be carried out`

For Example, you can create stages in the pipeline based on a specific type of branch
```
stage('fix branch){
  when{
    branch "fix-*"
  }
  steps{
    sh ```
    cat README.md
    ```
  }
}

stage('merge pr){
  when{
    branch "pr-*"
  }
  steps{
    echo "this is for prs"
  }
}
```

5. `Dealing with Pull Requests`: Now you need to create a PR to merge this branch back into the `main` which will in this scenario also update the root jenkins file so going forward each type of branch will have a specific pathway
      - Pull Requests also show up in the Jenkins build page for the multibranch pipeline
      - **Any PR or branch that has a strikethough means that it was already `merged and then the branch deleted` so it no longer exists** These go away next time a scan occurs