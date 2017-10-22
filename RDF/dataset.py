
file1=open('RDF/completeDataset.tsv','r')
file2=open('dataset/aifb.dataset', 'w')
file3=open('node/aifb.node', 'r')

file1.readline()#ignore the first line
data=file1.readlines()
nodes=file3.readlines()
for line in data:
	array=line.split('\t')
	klasse=array[2].split('\r')
	#node_to_find="\'{}\'".format(array[1])
	node_to_find=array[1]
	print node_to_find
	for nod in nodes:
		if node_to_find in nod:
			k_node=nod.split(' ')[0]
			towrite= "{} {}\n".format(k_node, klasse[0])
			file2.write(towrite)
			break
file1.close()
file2.close()
file3.close()
