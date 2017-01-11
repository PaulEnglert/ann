# Artificial Neural Networks

This repository is an playground to implement a full-flexed artificial neural network in python.

**Currently Implemented Network Topologies:**

* single-layer feed-forward neural network with dynamic numbers of neurons
* multi-layer feed-forward neural network (using backpropagation) with dynamic numbers of layers and neurons
* restricted boltzmann machine with dynamic numbers of hidden neurons


## Development Notes
* install python requirements in virtualenvironment based on `requirements.txt`.
* docs are created by running `make html` inside `doc/` directory
* tests are run by executing `nosetests` in the root of the project (use `nosetests --nocapture` to also see print statements)