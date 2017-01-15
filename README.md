# --------------------------
# --------------------------

# THIS REPO IS NOT MAINTAINED ANYMORE - CHECK [ML](https://github.com/PaulEnglert/ML) FOR NEWEST UPDATES

# --------------------------
# --------------------------



# Artificial Neural Networks

This repository is an playground to implement a full-flexed artificial neural network in python.

**Currently Implemented Network Topologies:**

* single-layer feed-forward neural network with dynamic numbers of neurons
* multi-layer feed-forward neural network (using backpropagation) with dynamic numbers of layers and neurons
* restricted boltzmann machine with dynamic numbers of hidden neurons


## Development Notes
* install python requirements in virtualenvironment based on `requirements.txt`.
* docs are created by running `make html` inside `doc/` directory
* tests are run by executing `nosetests` in the root of the project (use `nosetests --nocapture` to also see print statements, to run a single test use e.g. `nosetests tests.test_advanced:AdvancedTestSuite.test_restricted_boltzmann_machine_nb`)

Bugs:

* for `bigfloat` package do: `brew install gmp; brew install mpfr; pip install --global-option=build_ext --global-option="-I/usr/local/include" --global-option="-L/usr/local/lib" bigfloat
* for `opencv` aka `cv2` install opencv with homebrew and copy the python libs `cv.py` and `cv2.so` to the virtual environment site-packages directory