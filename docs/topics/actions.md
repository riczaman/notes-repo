Platform to automate Developer Workflows. CI/CD pipelines is just one of the many workflows that GitHub Actions can be used for.
- *Any GitHub event that occurs can trigger a workflow which is an automated execution of tasks = GitHub Action*

---

##CI&CD Pipelines

1. commit code
2. test code
3. build code
4. push artifact
5. deploy artifact on server

---

##GitHub Action CI/CD Example

- GitHub Actions already has a bunch of workflow templates for applications. 3 types of workflows: Deployment, continious integration build and test, and continious integration publish workflows 

- If you go to `Actions` on your repo and then select a `workflow` as a template than it will automatically create the following:
   - `.github/workflows`: Creates the folder that holds the Action workflow
   - `*.yml`: The yaml file that contains the instuctions

---

##Workflow Syntax

Main Parameters include:
```
   - `name`: This is just the name of the Action
   - `on`: This is the section that describes the event you can also do multiple events like *on and pull request*
        `push`: 
            `branches`: [Master]
    
    - `jobs`: A set of actions that get executed 

        -  `runs-on`: ubuntu-latest

        - `steps`: 
           - `uses`: actions/checkout@v2 - This is a predefined action that handles your repo checkout that was made my GitHub. If you go to github.com/actions you can see the full list of pre-defined actions 

           - name: Set up JDK 1.8
                uses: actions/setup-java@v1
                with:
                    java-version: 1.8

             - name: Grant execute permission for gradlew
                run: chmod +x gradlew    #Whenver we use an action we use the *uses* keyword but when we need to run a specific command use *run*
```
    - When you push to master or make a PR to master that above action will automatically start
   
    - The code is run on GitHub servers which means you do not have to manage them. Each new job runs on a new virtual enviornment. In the above example, we only put one job but you can put multiple jobs and they will all run on different virtual machines.
       - **By default the jobs run in parallel but you can overwrite this by using the keyword needs under the second jobs like `needs: firstjob`**
    
    - The runs on command determines what OS to run the build on (ubuntu, macOS, or Windows). Using the `matrix` keyword is needed when you want to test on multiple OS then you switch the runs-on to use {{matrix.os}}

---

## Build a Docker Image from the Artifact Generated

- You define this also within the workflow action as another step

```
    - name: Build and push Docker Image
        run: You can put all of the commands required to build and then bush the docker image and if you need multiple commands then you use the pipe syntax. ie. |
        docker login cred
        docker build
            - On Ubuntu machines docker is already pre-installed so you don't have to setup docker to use the commands
            - but instead of using the run command there is an action for the docker build and push and these are found in the *GitHub Actions Marketplace*

```
- **For credentials you can store them in GitHub as secrets**


