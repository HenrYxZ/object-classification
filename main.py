import pickle

#--------------------------
# Local dependencies
from dataset import Dataset

def main():
	dataset_path = "dataset"
	dataset = Dataset(dataset_path)
	dataset.generate_sets()
	dataset.store_listfile()
	pickle.dump(dataset, open("dataset.obj", "wb"), protocol = 2)

if __name__ == '__main__':
	main()