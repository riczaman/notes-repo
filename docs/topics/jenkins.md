## Jenkins Notes

Jenkins is an open-source automation server used for continuous integration and delivery (CI/CD).
<pre> python def hello(): print("Hello, Jenkins!")</pre>
```oo```

<pre>bash<br>docker pull jenkins/jenkins:lts<br></pre> 

!!! note "Why Docker is useful here"

    Running Jenkins in Docker avoids messing with your local setup.

    ```bash
    docker pull jenkins/jenkins:lts
    docker run -p 8080:8080 -p 50000:50000 jenkins/jenkins:lts
    ```

---

## Overview
- Automates building, testing, and deploying software.
- Highly customizable via plugins.
- Supports pipelines written in **Groovy**.

---

## Installation

!!! note "Why Docker is useful here"
    Running Jenkins in Docker avoids messing with your local setup.

```bash
docker pull jenkins/jenkins:lts
docker run -p 8080:8080 -p 50000:50000 jenkins/jenkins:lts

Accessing Jenkins
Open your browser: http://localhost:8080

Retrieve the admin password:

bash
Copy
Edit
docker exec -it <container_id> cat /var/jenkins_home/secrets/initialAdminPassword
Example Pipeline
??? example "Click to view a basic pipeline"
groovy pipeline { agent any stages { stage('Build') { steps { echo 'Building...' } } stage('Test') { steps { echo 'Running tests...' } } stage('Deploy') { steps { echo 'Deploying...' } } } }

Screenshot

Key Points
Keep pipelines modular.

Use environment variables for secrets.

Test locally before pushing changes.

yaml
Copy
Edit

---

### Folder Structure
docs/
index.md
jenkins.md
images/
jenkins-dashboard.png

markdown
Copy
Edit

---

If you want, I can now give you a **full starter repo** with:
- `mkdocs.yml` preconfigured for Material theme
- `index.md`
- `jenkins.md`
- `images/` folder
- GitHub Action ready for deployment

That way you just push and see it live.  
Do you want me to set that up?