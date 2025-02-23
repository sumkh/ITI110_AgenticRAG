# Build the Gradio App Image

* Create a new directory named **app*.
* This new directory should only contain (mandatory):
  - Dockerfile (no file extension)
  - your application code (app.py)
  - requirement.txt
* Other artifacts that your application required (e.g. vector databases) should also be saved in the *app* folder. Be care not to include other files you do not need to be copied into the docker images.
* Finally, `cd app` to the **app** directory and run the following command to build the docker image. 

```bash
docker build -t my-gradio-app .
```
This command creates an image tagged my-gradio-app.


## Example of Dockerfile:
```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port on which Gradio will run (default is 7860)
EXPOSE 7860

# Define environment variable for the vLLM endpoint (adjust as needed)
# This could also be passed in at runtime via docker run -e VLLM_ENDPOINT=...
ENV VLLM_ENDPOINT=http://vllm-instance:8000

# Run the application
CMD ["python", "app.py"]
```

### Output from Bocker Build
```bash
(yourenv) (base) briansum@Brians-MacBook-Pro app % docker build -t my-gradio-app .

[+] Building 102.7s (11/11) FINISHED                                                                docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                                0.0s
 => => transferring dockerfile: 638B                                                                                0.0s
 => [internal] load metadata for docker.io/library/python:3.9-slim                                                  2.1s
 => [auth] library/python:pull token for registry-1.docker.io                                                       0.0s
 => [internal] load .dockerignore                                                                                   0.0s
 => => transferring context: 2B                                                                                     0.0s
 => [1/5] FROM docker.io/library/python:3.9-slim@sha256:f9364cd6e0c146966f8f23fc4fd85d53f2e604bdde74e3c06565194dc4  0.0s
 => [internal] load build context                                                                                   0.1s
 => => transferring context: 11.48MB                                                                                0.1s
 => CACHED [2/5] WORKDIR /app                                                                                       0.0s
 => [3/5] COPY requirements.txt .                                                                                   0.0s
 => [4/5] RUN pip install --no-cache-dir -r requirements.txt                                                       95.7s
 => [5/5] COPY . .                                                                                                  0.0s
 => exporting to image                                                                                              4.7s
 => => exporting layers                                                                                             4.7s
 => => writing image sha256:91f200bb4c4475805785dd71b004c61476ca9fc36e063e74a99e2dffff94d6a1                        0.0s
 => => naming to docker.io/library/my-gradio-app                                                                    0.0s

View build details: docker-desktop://dashboard/build/desktop-linux/desktop-linux/lsisitnbdebjtmrb222mh8c6u

What's next:
    View a summary of image vulnerabilities and recommendations → docker scout quickview 
```

## To update your Docker image after making changes to source files, 
-> need to rebuild the image.

### 1. Rebuild the Image

Simply run the build command again from your project’s root directory:

```bash
docker build -t my-gradio-app .
```

This command rebuilds the image using the updated files. Docker uses caching to speed up the build process, so if want to ensure all steps are executed fresh, can disable the cache:

```bash
docker build --no-cache -t my-gradio-app .
```

### 2. Restart the Container

After rebuilding, will need to stop (if already run) and remove the old container, then run a new one using the updated image:

```bash
docker stop gradio-chatbot
docker rm gradio-chatbot
docker run -d --name gradio-chatbot --network my-network -p 7860:7860 my-gradio-app
```

### 3. Using Docker Compose (Optional)

If using Docker Compose, can rebuild and restart services with:

```bash
docker-compose up --build
```

This command rebuilds images and starts the containers according to `docker-compose.yml`.