# Docker Python-ML
Dockerised Python Sandbox for ML

### Prerequisites:
    docker engine
    docker compose

### Getting started

 I wanted to keep all python dependencies in a container instead of installing to my host machine so: `docker-compose -f development.yml up -d` Will build my image and install whats needed, simple

To execute Python scripts:
`docker exec python-ml python3.6 entrypoint.py`

To Start Cats vs Dogs Process
`docker exec python-ml python3.6 cats-vs-dogs.py`

To Enter the Container
`docker exec -it python-ml bash`

You can use Matplotlib to `.show()` your result output but because i use docker i could be bothered setting up an X server and handling display so i just save output image to `results/images`
