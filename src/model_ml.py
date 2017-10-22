import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score

def parse_args():
	'''
	Parses the file argument.
	'''
	parser = argparse.ArgumentParser(description="Run parse.")

	parser.add_argument('--input', nargs='?', default='emb/gewicht.emb',help='Input emb path')
	parser.add_argument('--dataset', nargs='?', default='dataset/gewicht.dataset',help='Input dataset path')
	parser.add_argument('--y_true', nargs='?', default='dataset/gewicht.true',help='Input y_true prediction path')
	return parser.parse_args()


if __name__ == '__main__':
	args=parse_args()
	file_input= open(args.input, "r") 
	data_input=file_input.readlines()

	file_dataset= open(args.dataset, "r") 
	data_dataset=file_dataset.readlines()

	file_y_true= open(args.y_true, "r")
	data_y_true=file_y_true.readlines()

	node_feature={}
	node_dataset={}
	node_to_predict=[3549,2387,165,1007,1938,1045,1123,299,278,772,1266,3152,3192,1168]#,1097,1458,167,3349,136,201]
	y_predict=[]
	y_true=[]

	#build a dictionary of features
	for i in range(1,len(data_input)):
		array=data_input[i].split(" ")
		key=array[0]
		value=[]
		for j in range(1,len(array)):
			value.append(float(array[j]))
		node_feature[key]=value
	#print node_feature

	#build a dictionary of dataset
	for line in data_dataset:
		array=line.split(' ')
		#node_dataset[array[0]]=int(array[1])
		node_dataset[array[0]]=str(array[1].split('\n')[0])

	#load the true prediction in an array
	for line in data_y_true:
		array=line.split(' ')
		#node_dataset[array[0]]=int(array[1])
		y_true.append(str(array[1].split('\n')[0]))

	X=[]
	Y=[]
	for node, value in node_dataset.iteritems():
		X.append(node_feature[node])
		Y.append(value)
	print X[0]
	print Y
	clf = tree.DecisionTreeClassifier(random_state=0)
	clf = clf.fit(X, Y)

	for node_key in node_to_predict:
		predicted=clf.predict([node_feature[str(node_key)]])
		y_predict.append(predicted[0])

	#print y_predict
	#print y_true

	print "accuracy_score = {}".format(accuracy_score(y_true, y_predict))
	#print clf.predict([[0.001348,0.000335,-0.003266,-0.001099,0.001716,-0.001004,-0.003479,0.001890,0.002609,0.003880,-0.000768,0.002727,0.003074,-0.000325,0.000982,-0.003332,-0.001377,0.001638,-0.002948,0.003709,0.003799,-0.000506,0.003869,0.000583,0.003273,0.003373,0.000641,-0.000103,0.001774,0.001243,-0.000032,0.002395,0.001400,0.002414,0.001845,-0.001972,-0.001318,-0.000368,0.001114,0.003171,-0.001206,0.003774,-0.003205,-0.001432,-0.000895,0.002802,-0.003499,0.000367,0.001640,-0.001930,-0.000342,-0.003123,0.003120,0.003416,-0.000569,0.003514,0.000669,-0.003597,-0.001848,-0.001215,0.001803,-0.003585,0.000115,-0.000617,0.003234,0.000785,0.000695,-0.002014,0.003254,0.003472,-0.002413,-0.000402,0.002720,-0.003676,0.002938,0.002300,0.003514,0.002240,0.001484,-0.000519,-0.003621,0.002254,0.002225,-0.002833,-0.003103,0.001434,0.002863,-0.000335,-0.002711,0.002904,0.000531,-0.002884,0.003121,-0.003125,-0.003904,0.001456,-0.002130,0.002909,0.001692,-0.001949,-0.000683,0.002776,0.000004,0.001010,0.000534,0.000440,-0.002839,-0.003244,-0.000908,-0.003025,-0.002278,-0.002622,-0.002911,-0.002181,0.000490,-0.002442,-0.000053,-0.003157,0.002946,-0.001108,-0.000569,-0.003412,-0.000700,-0.001080,0.002265,-0.003749,-0.003042,-0.002423]])
	#print clf.predict([[0.003389,-0.001924,0.003413,-0.001977,-0.000992,0.001033,-0.000975,-0.002369,0.002027,0.000342,-0.003190,0.002379,-0.001126,0.000938,-0.001332,0.001073,-0.003043,-0.003241,0.002558,0.003518,0.001096,-0.003243,0.000235,0.001814,0.000060,0.003224,-0.001053,0.000450,-0.001285,-0.001385,0.000124,0.001254,0.001166,-0.003831,-0.001030,-0.000083,-0.003853,0.002486,0.002652,0.001136,0.000341,-0.000671,-0.000101,-0.001399,0.003158,0.001591,0.002537,-0.002652,-0.000878,0.001461,-0.001461,-0.000739,-0.003355,-0.001789,0.003511,0.002377,-0.002960,0.001912,0.003745,-0.001236,-0.000643,-0.002271,0.003081,0.002428,-0.000448,-0.003280,0.002852,-0.000865,-0.002229,-0.002395,0.000819,0.000000,0.000608,0.000801,-0.002168,-0.003751,0.002877,0.002152,-0.003191,-0.002951,-0.000789,0.000431,0.000272,-0.003021,-0.002080,0.000084,0.002753,0.001717,-0.002885,-0.001540,-0.000586,0.002752,0.003759,0.002152,-0.001952,0.002946,-0.000558,0.000428,0.000664,-0.002841,0.002122,0.002155,-0.003016,-0.002471,-0.000880,0.000425,-0.000513,0.000227,-0.000842,-0.001219,-0.002086,-0.000195,0.000713,-0.002274,0.003106,0.002475,0.003658,0.003104,-0.002833,0.002896,-0.001251,-0.001281,-0.001854,-0.002614,-0.003277,0.000821,-0.003498,-0.002690]])
	
	#plt.plot(X)
	#plt.scatter(X, Y)
	#plt.show()