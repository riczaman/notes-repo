Docker is an open platform designed for developing, shipping, and running applications using containerization. At a high level, it enables users to package an application and all its dependencies (code, libraries, system tools, runtime, and configurations) into a standardized unit called a container. 

## Deployment before Containers
- Each developer needs to install the services and dependencies locally on their OS on their laptop (Redis, Database, Messaging, etc)
- Installation on different OS (Mac vs Windows) is different
- Prone to error as the setup can be difficult especially if your application is complex

## Deployment with Containers
- With containers like Docker you don't have to install all of the services independently on your OS instead these services come packaged with the application in an isolated enviornment
- Docker also allows you to run different versions of an application without any conflicts (ie. Redis 4.1, Redis 4.2, Redis 4.3, etc)

## VM vs Docker
- Docker only contains and deals with the OS application layer
- VM has the container layer and the OS Kernel
- *As a result Docker packages are a lot smaller since they only have to implement one layer of the OS and they are faster to start up since VMS have to boot up the OS Kernel*
- **VMs are compatible with all OS but Docker is only compatible with Linux OS for example if you have Linux Docker file it can run on a Windows machine because the Windows machine has the Windos OS Kernel**
- `Docker Desktop` now lets Linux Containers work with Mac or Windows OS. Docker Desktop uses a hypervisor layer which has the Linux kernel that is needed. 

---

## Docker Images vs Docker Containers

- `Docker image` is the artifact that we create that has the application and all of it envionrment dependencies that we can then upload to an artifact repository so another server/person can download and run it & it is executable

- `Docker Container` is a running instance of an image. Basically the server or host or machine that runs the image becomes the Docker container 
            - *You can run multiple containers from one image*

- `docker images`: **Command to see all the images that you have in your docker instance**
- `docker ps`: **See your running containers**

---

## Docker Registry

- Registries are places that have premade docker images stored
- `Docker Hub` is Docker's officially registry: https://hub.docker.com/

---

## Image Versioning

- New versions of the application require new versions of the docker image and these versions are called **Tags**
- All images have a tag called `latest` which indicates it the newest version of that image. If you don't choose a specific version you get latest by default

---

## Getting Images

1. Search for the image (package) that you need from a Registry
2. Then run `docker pull {name}:tag`
   - You didn't have to specify the registry here because DockerHub is the default location where it searches. 

![Docker Pull GIF](/notes-repo/images/dockerpull.gif)

---

## Running Images

1. To run images you just need to use this commmand: `docker run imageName:tagnNumber`
   - Running the container using this command runs it in the foreground which means your terminal is now blocked

2. Using `-d` flag will run the container in the background and it will give you the full process ID
   - ` docker run -d name:tag` 

3. To see **Logs** while running containers in the backround: `docker logs containerID` (you get this from docker ps)

4. *You can skip the docker pull command and just run the docker run for an image that you don't have locally as long as that image is found in DockerHub*

---

# Port Binding
