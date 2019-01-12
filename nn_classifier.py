# Leonard R. Kosta Jr.

import h5py
import tensorflow as tf
import numpy as np

import make_dataset

DEFAULT_DATASET_NAME_TRAIN = 'data_train.hdf5'
DEFAULT_DATASET_NAME_TEST = 'data_test.hdf5'
DEFAULT_EPOCHS = 20


def evaluate_model(model, x_test, y_test):
    """Evaluates the model and prints results."""
    results = model.evaluate(x_test, y_test)
    print('Results on the test dataset:')
    print('Loss: {0}\nAccuracy: {1}'.format(results[0], results[1]))
    logits = model.predict(x_test)
    y_pred = np.array([0 if p[0] >= 0.5 else 1 for p in logits])
    tp = sum([1 for i in range(len(y_test))
        if y_pred[i] == 1 and y_test[i] == 1])
    fp = sum([1 for i in range(len(y_test))
        if y_pred[i] == 1 and y_test[i] == 0])
    tn = sum([1 for i in range(len(y_test))
        if y_pred[i] == 0 and y_test[i] == 0])
    fn = sum([1 for i in range(len(y_test))
        if y_pred[i] == 0 and y_test[i] == 1])
    print('True positives: {0}'.format(tp))
    print('False positives: {0}'.format(fp))
    print('True negatives: {0}'.format(tn))
    print('False negatives: {0}'.format(fn))


def train_model(model, x_train, y_train, epochs=DEFAULT_EPOCHS):
    """Trains the model."""
    model.fit(x_train, y_train, epochs=epochs)


def get_model():
    """Returns the model. The model is an instance of
    tf.keras.models.Model (or one of its subclasses)."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_dataset(infile):
    """Returns the dataset at infile as an (x, y) tuple. Each is an
    np.array."""
    x = None
    y = None
    with h5py.File(infile, 'r') as f:
        x = f[make_dataset.DEFAULT_DATA_KEY][()]
        y = f[make_dataset.DEFAULT_LABEL_KEY][()]
    return (x, y)


def main():
    """Runs the program."""
    x_train, y_train = get_dataset(DEFAULT_DATASET_NAME_TRAIN)
    x_test, y_test = get_dataset(DEFAULT_DATASET_NAME_TEST)
    model = get_model()
    train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)


if __name__ == '__main__':
    main()

