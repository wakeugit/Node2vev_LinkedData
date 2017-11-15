import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from rdflib import OWL
from rdflib import Graph
import main
from sklearn.model_selection import cross_val_score
from random import randint

def parse_args():
	'''
	Parses the file argument.
	'''

	parser = argparse.ArgumentParser(description="Run parse.")
	parser.add_argument('--graph', nargs='?', default='graph/aifb_fixed_complete.nt',help='graph path')

	parser.add_argument('--dataset', nargs='?', default='graph/aifb_completeDataset.tsv',help='dataset path')

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=500,
	                    help='Number of dimensions. Default is 128.') #dimensions d

	parser.add_argument('--walk-length', type=int, default=8,
	                    help='Length of walk per source. Default is 80.') #length of walk l

	parser.add_argument('--num-walks', type=int, default=500,
	                    help='Number of walks per source. Default is 10.') #walks per node r

	parser.add_argument('--window-size', type=int, default=5,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=10, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.') #p > max(p,1) to avoid edundancy

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--labeled', dest='labeled', action='store_true',
	                    help='Boolean specifying (un)labeled. Default is labeled.')
	parser.add_argument('--unlabeled', dest='unlabeled', action='store_false')
	parser.set_defaults(labeled=True)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is directed.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=True)

	return parser.parse_args()
	return parser.parse_args()

def edgelist(graph):
	'''
	From a graph use rdf to generate the edgelist"
	'''
	g = Graph()
	arr=graph.split('.')
	if arr[len(arr)-1]=='nt':
		g.parse(graph, format="nt")
	else:
		g.parse(graph)
	
	out_file= graph.split('.')[0]+'.edgelist'
	
	f = open(out_file, 'w')

	g1=g.subject_objects()
	out_file= graph.split('.')[0]+'.edgelist'
	f = open(out_file, 'w')

	class_to_remove=["affiliation", "lithogenesis"]
	found=False

	for (s,p,o) in g:
		type_of_subject=str(type(s))
		type_of_object=str(type(o))
		found=False
		if ('Literal' not in type_of_object) and ('Literal' not in type_of_subject):
			for c in class_to_remove:
				if c in p.encode('utf-8'):
					found=True
					break
			if found==False:
				towrite= "{} {} {}\n".format(s.encode('utf-8'), o.encode('utf-8'), p.encode('utf-8'))
				f.write(towrite)
			else:
				found=False
	f.close()
	return out_file
'''
	for (s,o) in g1:
		type_subject=str(type(s))
		type_object=str(type(o))
		if ('Literal' not in type_object) and ('Literal' not in type_subject)  :
			towrite= "{} {}\n".format(s.encode('utf-8'), o.encode('utf-8'))
			f.write(towrite)
	f.close()
	return out_file
'''

	


def pca_plot(X):
	print len(X)
	pca = PCA(n_components=2)
	pca.fit(X)
	X=pca.transform(X)
	print X
	
	plt.plot(X[:,0], X[:,1], 'bo')
	
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()


if __name__ == '__main__':
	args=parse_args()
	#generate the edgelist
	args.input=edgelist(args.graph)
	args.output='emb/'+args.input.split('.')[0].split('/')[1]+'.emb'
	print args.output
	
	node_dataset={}
	node_feature={}
	node_to_plot=[]

	#generate the embeddings 
	main.generate_embeddings(args)

	#build a dictionary of features
	file_emb= open(args.output, "r") 
	data_emb=file_emb.readlines()

	for i in range(1,len(data_emb)):
		array=data_emb[i].split(" ")
		key=array[0] 		#node
		#key=array[0].split('@')[0] 		#node
		value=[]			#embeddings
		for j in range(1,len(array)):
			value.append(float(array[j]))
		node_feature[key]=value

	#build a dictionary of person + classification
	file_dataset=open(args.dataset)
	file_dataset.readline()	#read the first line
	dataset=file_dataset.readlines()

	for data in dataset:
		array=data.split('\t')
		#key=array[0]				#person
		#value=array[1].split('\r')[0]	#classification
		key=array[1]				#person
		value=array[2].split('\r')[0]	#classification
		node_dataset[key]=value


	#array of node to plot
	for p in node_dataset.keys():
		node_to_plot.append(node_feature.get(p))
	
	X=[]

	for elt in node_dataset.keys():
		X.append(node_feature.get(elt))

	dt = tree.DecisionTreeClassifier(random_state=0)
	csvm = svm.SVC(kernel='linear', C=1)
	gnb = GaussianNB()
	KNN = KNeighborsClassifier(n_neighbors=3)


	scores_dt=cross_val_score(dt, X, node_dataset.values(), cv=10)
	#print scores_dt
	durchnitt=0
	for s in scores_dt:
		print s
		durchnitt=durchnitt+s/len(scores_dt)
	print "accuracy_score_C4.5 = {}".format(durchnitt)

	scores_csvm=cross_val_score(csvm, X, node_dataset.values(), cv=10)
	#print scores_csvm
	durchnitt=0
	for s in scores_csvm:
		print s
		durchnitt=durchnitt+s/len(scores_csvm)
	print "accuracy_score_SVM = {}".format(durchnitt)

	scores_KNN=cross_val_score(KNN, X, node_dataset.values(), cv=10)
	#print scores_KNN
	durchnitt=0
	for s in scores_KNN:
		print s
		durchnitt=durchnitt+s/len(scores_KNN)
	print "accuracy_score_KNN = {}".format(durchnitt)

	scores_gnb=cross_val_score(gnb, X, node_dataset.values(), cv=10)
	#print scores_gnb
	durchnitt=0
	for s in scores_gnb:
		print s
		durchnitt=durchnitt+s/len(scores_gnb)
	print "accuracy_score_NB = {}".format(durchnitt)

	

	'''
	**********************************************************************************************************************************
		in this part we plot the entities of the dataset

	'''

	colors = {}
	color=['red', 'green', 'blue','white','orange','black','yellow', 'gray', 'purple']
	j=0
	visited_group=[]
	group=node_dataset.values()		
	X_plot=[]

	pca = PCA(n_components=2)
	pca.fit(node_to_plot)
	X_plot=pca.transform(node_to_plot)
	
	for i in range(len(X_plot)):
		if group[i] not in visited_group:
			visited_group.append(group[i])
			colors[group[i]]=color[j]
			j=j+1
		plt.scatter(X_plot[i,0], X_plot[i,1], c=colors.get(group[i]))


	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()