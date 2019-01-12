default_train_dataset_name = data_train.hdf5
default_test_dataset_name = data_test.hdf5
default_train_plot_name = data_plot_train.png
default_test_plot_name = data_plot_test.png
default_train_dataset_size = 10000
default_test_dataset_size = 1000
default_num_features = 2

all: run

run:
	echo "Assumes that the dataset has already been made (i.e. make dataset)!"
	echo "Trying default python installation:"
	python nn_classifier.py || (echo "On Mac OS X, matplotlib needs to use a framework build; retrying."; pythonw nn_classifier.py)

dataset: dataset_train dataset_test

dataset_train:
	echo "Trying default python installation:"
	python make_dataset.py --n $(default_train_dataset_size) --m $(default_num_features) --outfile $(default_train_dataset_name) --plot_file $(default_train_plot_name) || (echo "On Mac OS X, matplotlib needs to use a framework build; retrying."; pythonw make_dataset.py --n $(default_train_dataset_size) --m $(default_num_features) --outfile $(default_train_dataset_name) --plot_file $(default_train_plot_name))

dataset_test:
	echo "Trying default python installation:"
	python make_dataset.py --n $(default_test_dataset_size) --m $(default_num_features) --outfile $(default_test_dataset_name) --plot_file $(default_test_plot_name) || (echo "On Mac OS X, matplotlib needs to use a framework build; retrying."; pythonw make_dataset.py --n $(default_test_dataset_size) --m $(default_num_features) --outfile $(default_test_dataset_name) --plot_file $(default_test_plot_name))

clean:
	rm $(default_train_dataset_name)
	rm $(default_test_dataset_name)
	rm $(default_train_plot_name)
	rm $(default_test_plot_name)