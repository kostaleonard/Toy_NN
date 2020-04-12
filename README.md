# toy_nn

This simple project is set up with a basic neural network and dataset generator in order to familiarize users with Tensorflow and machine learning concepts. Makefile included.

## Running instructions

Clone the repository.
```git clone https://kostaleonard@bitbucket.org/kostaleonard/toy_nn.git
```

On systems that already have TensorFlow and all dependencies installed (probably NOT your machine), build the dataset and then run the neural network.
```make dataset
make
```

On any other system (probably your machine!), run the project in a Docker container.
```make docker
make
```

