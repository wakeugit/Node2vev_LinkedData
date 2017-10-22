import argparse
from rdflib import Graph


def parse_args():
	'''
	Parses the file argument.
	'''
	parser = argparse.ArgumentParser(description="Run parse.")

	parser.add_argument('--graph', nargs='?', default='RDF/aifb_fixed_complete.nt',help='Input graph path')
	parser.add_argument('--node_out', nargs='?', default='node/aifb.node',help='Input node-output path')
	parser.add_argument('--edgelist', nargs='?', default='graph/aifb.edgelist',help='Input edgelist output path')
	return parser.parse_args()

if __name__ == '__main__':
	g = Graph()
	#args=parse_args
	g.parse('RDF/aifb_fixed_complete.nt', format="nt")
	#g.parse("http://www.w3.org/People/Berners-Lee/card")
	#g.parse("RDF/aifb_fixed_complete.nt", format="nt")
	g1=g.subject_objects()
	g2=g.subject_objects()
	node=[]

	
	f = open('graph/aifb2.edgelist', 'w')

	for (s,o) in g1:
		type_subject=str(type(s))
		type_object=str(type(o))
		if ('Literal' not in type_object) and ('Literal' not in type_subject)  :
			towrite= "\'{}\';;\'{}\'\n".format(s.encode('utf-8'), o.encode('utf-8'))
			f.write(towrite)
	f.close()
	#print node

	
'''

	f = open(args.node_out, 'w')
	#f = open('node/aifb.node', 'w')
	i=1
	for (s,o) in g1:
		if s not in node.keys():
			node[s]=i
			i=i+1
			towrite= "{} \'{}\'\n".format(i, s.encode('utf-8'))
			f.write(towrite)
		if o not in node.keys():
			node[o]=i
			i=i+1
			towrite= "{} \'{}\'\n".format(i, o.encode('utf-8'))
			f.write(towrite)
	f.close()


	file = open(args.edgelist, 'w')
	#file = open('graph/aifb.edgelist', 'w')
	for (s,o) in g2:
		towrite= "{} {}\n".format(node.get(s), node.get(o))
		file.write(towrite)
	file.close() 

'''