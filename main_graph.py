import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import seaborn as sns
import pickle
import pdb
import json
import networkx as nx

def load_csv(filename='building_data.csv'):
	df = pd.read_csv(filename)
	return df

def parse_node_graph():
	
	abc = json.load(open('brick_relationhsips.json', 'r'))
	graph = nx.DiGraph()
	nodes = dict()
	i = 0

	for key, val in abc['hasLocation'].items():
		
		if 'Node' in key and 'Temperature' in key:
			i += 1
			print(f'{key[:-12].upper()} -> {val.upper()}')
			nodes[key[:-12].upper()] = val.upper()

	pickle.dump(nodes, open('nodes.pkl', 'wb'))
	pdb.set_trace()

def parse_vav_graph():
	
	abc = json.load(open('brick_relationhsips.json', 'r'))
	graph = nx.DiGraph()
	points = set()

	for key, val in abc['isFedBy'].items():
		points.add(key.upper())
		points.add(val.upper())

	graph.add_nodes_from(points)

	for key, val in abc['isFedBy'].items():
		graph.add_edge(val.upper(), key.upper())
	
	for node in graph.nodes():
		for item in graph[node]:
			print(f'{node} -> {item}')

	nx.write_gml(graph, f"brick-graph.gml")
	plot_graph(graph)

def plot_graph(graph, positions=None, name='brick-graph.pdf'):
	
	node_list = list(graph.nodes)

	if positions is None:
		positions = nx.planar_layout(graph)
		
	nx.draw_networkx_edges(graph, positions, arrows=True)
	nx.draw_networkx_labels(graph, positions, font_size=4)
	nx.draw_networkx_nodes(graph, positions, nodelist=node_list)

	plt.axis('off')
	plt.savefig(name)
	plt.close()

	return 

if __name__ == '__main__':
	parse_node_graph()