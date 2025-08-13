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

