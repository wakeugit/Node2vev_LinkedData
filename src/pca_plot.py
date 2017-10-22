import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA

def parse_args():
	'''
	Parses the file argument.
	'''
	parser = argparse.ArgumentParser(description="Run parse.")

	parser.add_argument('--input', nargs='?', default='emb/aifb.emb',help='Input emb path')
	return parser.parse_args()

if __name__ == '__main__':
	X=[]
	args=parse_args()
	file_input= open(args.input, "r") 
	data_input=file_input.readlines()
	person=[]
	node_feature={}

	#person in dataset
	file=open("RDF/completeDataset.tsv", "r")
	file.readline()
	dataset=file.readlines()
	for line in dataset:
		print line
		person.append(line.split('\t')[1])
	print person


	
	#build a dictionary of features
	for i in range(1,len(data_input)):
		array=data_input[i].split(" ")
		key=array[0]
		value=[]
		for j in range(1,len(array)):
			value.append(float(array[j]))
		node_feature[key]=value
		#X.append(value)
	print node_feature
	for p in person :
		print node_feature.get(p)

	print len(X)
	pca = PCA(n_components=2)
	pca.fit(X)
	X=pca.transform(X)
	print X
	plt.plot(X[:,0], X[:,1], 'bo')
	plt.xlabel('x1')
	plt.ylabel('x2')
	#pl.xlim(0.0, 10.)
	#plt.plot(X)
	plt.show()

	