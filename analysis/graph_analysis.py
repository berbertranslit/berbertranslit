import pandas as pd
from collections import Counter
import os
import sys
import ast

from igraph import Graph
import re

with open("transliterate/output/tri_alignments.txt") as f:
    lines = f.readlines()


def getStripped(word):
    if len(word) in [0, 1]:
        out = word
    else:
        out = re.sub("[aaieəou]", "", word[:-1])
        out = out + word[-1]
    return out

unique_gold_exc = []
unique_pred_3b = []
unique_pred_3c = []
for index, line in enumerate(lines):
    if (index) % 6 == 1:
        split = line.strip("\n").split(":")
        for utt_ind, utterance in enumerate(split):
            utterance = utterance.strip()
            if utt_ind == 0:
                words = utterance.split(" ")
                words = [getStripped(x) for x in words]
                unique_gold.extend(words)
            if utt_ind == 1:
                words = utterance.split(" ")
                unique_pred_3b.extend(words)
            if utt_ind == 2:
                words = utterance.split(" ")
                unique_pred_3c.extend(words)
    #if (index) % 6 == 1:
        #print(line)


unique_gold_counter = Counter(unique_gold)
unique_pred_3b_counter = Counter(unique_pred_3b)
unique_pred_3c_counter = Counter(unique_pred_3c)
shared_counter = set(unique_gold_counter & unique_pred_3b_counter & unique_pred_3c_counter)

shared_3b_counter = set(unique_gold_counter & unique_pred_3b_counter)
shared_3c_counter = set(unique_gold_counter & unique_pred_3c_counter)


len(unique_gold_counter)
len(unique_pred_3b_counter)
len(unique_pred_3c_counter)

shared_set = set(shared_counter)
shared_3b_set = set(shared_3b_counter)
shared_3c_set = set(shared_3c_counter)


unique_3b_set = set()
unique_gold_set = set(unique_gold_counter)
for word_3b in unique_pred_3b:
    if word_3b not in unique_gold_set:
        unique_3b_set.add(word_3b)

unique_3c_set = set()
for word_3c in unique_pred_3c:
    if word_3c not in unique_gold_set:
        unique_3c_set.add(word_3c)

union_counter = unique_gold_counter + unique_pred_3b_counter + unique_pred_3c_counter

len(unique_gold_counter)

import numpy as np

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import editdistance
import time

arr1 = np.array(list(union_counter.keys()))
arr2 = np.array(list(union_counter.keys()))


unique_3b_binar = []
unique_3c_binar = []
neither_binar = []
both_binar = []
shared_binar = []
shared_3b_binar = []
shared_3c_binar = []
for word in arr1:
    compare = str(word)
    if compare in shared_set:
        shared_binar.append(1)
    else:
        shared_binar.append(0)
    if compare in unique_3b_set:
        unique_3b_binar.append(1)
    else:
        unique_3b_binar.append(0)
    if compare in unique_3c_set:
        unique_3c_binar.append(1)
    else:
        unique_3c_binar.append(0)
    if compare not in unique_3b_set and compare not in unique_3c_set:
        neither_binar.append(1)
    else:
        neither_binar.append(0)
    if compare in unique_3b_set and compare in unique_3c_set and compare not in unique_gold_set:
        both_binar.append(1)
    else:
        both_binar.append(0)
    if compare in shared_3b_set:
        shared_3b_binar.append(1)
    else:
        shared_3b_binar.append(0)
    if compare in shared_3c_set:
        shared_3c_binar.append(1)
    else:
        shared_3c_binar.append(0)

def condition(x): return x > 0
unique_3b_binar_cond = condition(np.array(unique_3b_binar))
unique_3c_binar_cond = condition(np.array(unique_3c_binar))
neither_binar_cond = condition(np.array(neither_binar))
shared_binar_cond = condition(np.array(shared_binar))
shared_3b_binar_cond = condition(np.array(shared_3b_binar))
shared_3c_binar_cond = condition(np.array(shared_3c_binar))

both_binar_cond = condition(np.array(both_binar))

unique_3b_binar_indices = np.where(unique_3b_binar_cond)[0]
unique_3c_binar_indices = np.where(unique_3c_binar_cond)[0]
neither_binar_indices = np.where(neither_binar_cond)[0]
shared_binar_indices = np.where(shared_binar_cond)[0]
shared_3b_binar_indices = np.where(shared_3b_binar_cond)[0]
shared_3c_binar_indices = np.where(shared_binar_cond)[0]

both_binar_indices = np.where(both_binar_cond)[0]

arr = np.array([1, 2, 3, 4, 5])
bool_arr = arr > 2
output = np.where(bool_arr)[0]
print(output)

def getEditMatrix(x, y):
    return cdist(arr2.reshape(-1, 1), arr1.reshape(-1, 1), lambda x, y: editdistance.distance(x[0], y[0]))

for n in [len(arr1)]:
    arr1 = np.array(list(union_counter.keys())[:n])
    arr2 = np.array(list(union_counter.keys())[:n])
    start = time.time()
    matrix = getEditMatrix(arr1, arr2)
    matrix = np.where(matrix > 1, 0, matrix)
    end = time.time()
    print(n, end-start)

n = nx.convert_matrix.from_numpy_array(np.nan_to_num(matrix))
np.save("adj.npy", matrix)

adj_matr_pd = pd.DataFrame(matrix)
adj_matr_pd.columns = arr1
adj_matr_pd.set_index(arr1, inplace=True)
adj_matr_pd = adj_matr_pd.fillna(0)

import networkx as nx
import random
n = nx.convert_matrix.from_pandas_adjacency(adj_matr_pd)

k = 1000
sampled_nodes = random.sample(n.nodes, k)
sampled_graph = n.subgraph(sampled_nodes)


from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
from bokeh.transform import linear_cmap
from networkx.algorithms import community

degrees = dict(nx.degree(sampled_graph))
nx.set_node_attributes(sampled_graph, name='degree', values=degrees)

number_to_adjust_by = 5
adjusted_node_size = dict([(node, degree+number_to_adjust_by) for node, degree in nx.degree(sampled_graph)])
nx.set_node_attributes(sampled_graph, name='adjusted_node_size', values=adjusted_node_size)

communities = community.greedy_modularity_communities(sampled_graph)

from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8, Viridis256
# Create empty dictionaries
modularity_class = {}
modularity_color = {}
label_dict = {}
unique_3b_binar_dict = {}
unique_3b_color = {}
unique_3c_binar_dict = {}
unique_3c_color = {}
both_color = {}
all_color = {}

#Loop through each community in the network
for community_number, community in enumerate(communities):
    #For each member of the community, add their community number and a distinct color
    rand_num = random.randrange(1, 256)
    for name in community:
        modularity_class[name] = community_number
        modularity_color[name] = Viridis256[rand_num]
        unique_3b_binar_dict[name] = unique_3b_binar[name]
        unique_3c_binar_dict[name] = unique_3c_binar[name]

for index, value in enumerate(unique_3b_binar):
    if value == 1:
        color = Viridis8[0]
    elif value == 0:
        color = Viridis8[-1]
    unique_3b_color.setdefault(index, color)

for index, value in enumerate(unique_3c_binar):
    if value == 1:
        color = Viridis8[0]
    elif value == 0:
        color = Viridis8[-1]
    unique_3c_color.setdefault(index, color)

for index, (b_value, c_value)  in enumerate(zip(unique_3b_binar, unique_3c_binar)):
    if b_value == 1 and c_value == 1:
        color = Viridis8[0]
    elif value == 0:
        color = Viridis8[-1]
    both_color.setdefault(index, color)


for index, (b_value, c_value, s_value)  in enumerate(zip(unique_3b_binar, unique_3c_binar, shared_binar)):
    if s_value == 1:
        color = Oranges8[6]
    elif b_value == 1 and c_value == 1:
        color = Reds8[0]
    elif b_value == 1 and c_value == 0:
        color = Viridis8[0]
    elif b_value == 0 and c_value == 1:
        color = Viridis8[4]
    else:
        color = Viridis8[-1]
    all_color.setdefault(index, color)

name_dict = dict(enumerate(arr1, start=0))

nx.set_node_attributes(sampled_graph, name='modularity_class', values=modularity_class)
nx.set_node_attributes(sampled_graph, name='modularity_color', values=modularity_color)
nx.set_node_attributes(sampled_graph, name='3b_error', values=unique_3b_binar_dict)
nx.set_node_attributes(sampled_graph, name='3c_error', values=unique_3c_binar_dict)
nx.set_node_attributes(sampled_graph, name='3b_color', values=unique_3b_color)
nx.set_node_attributes(sampled_graph, name='3c_color', values=unique_3c_color)
nx.set_node_attributes(sampled_graph, name='both_color', values=both_color)
nx.set_node_attributes(sampled_graph, name='all_color', values=all_color)

nx.set_node_attributes(sampled_graph, name='name', values=name_dict)

from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges

#Choose colors for node and edge highlighting
node_highlight_color = 'white'
edge_highlight_color = 'black'

#Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
size_by_this_attribute = 'adjusted_node_size'
color_by_this_attribute = 'all_color'

#Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8
color_palette = Blues8

#Choose a title!
title = 'wow'
#Establish which categories will appear when hovering over each node
HOVER_TOOLTIPS = [
    ("Word", "@name"),
    ("Community", "@modularity_class"),
    ("all color", "$color[swatch]:all_color")
]

#Create a plot — set dimensions, toolbar, and title
plot = figure(tooltips = HOVER_TOOLTIPS,
              tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
            x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

sampled_graph
#Create a network graph object
# https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
network_graph = from_networkx(sampled_graph, nx.spring_layout, scale=10, center=(0, 0))

#Set node sizes and colors according to node degree (color as category from attribute)
network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=color_by_this_attribute)
#Set node highlight colors
network_graph.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)
network_graph.node_renderer.selection_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)

#Set edge opacity and width
network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)
#Set edge highlight colors
network_graph.edge_renderer.selection_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)
network_graph.edge_renderer.hover_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)

    #Highlight nodes and edges
network_graph.selection_policy = NodesAndLinkedEdges()
network_graph.inspection_policy = NodesAndLinkedEdges()

plot.renderers.append(network_graph)

show(plot)

nx.info(n)

nx.density(n.subgraph(np.hstack((neither_binar_indices,unique_3b_binar_indices))))
nx.density(n.subgraph(np.hstack((neither_binar_indices,unique_3c_binar_indices))))

any_error_indices = np.unique(np.concatenate([both_binar_indices,unique_3b_binar_indices,unique_3c_binar_indices]))
b_indices = np.unique(np.concatenate([unique_3b_binar_indices,shared_3b_binar_indices]))
c_indices = np.unique(np.concatenate([unique_3c_binar_indices,shared_3c_binar_indices]))

shared_density = nx.density(n.subgraph((shared_binar_indices)))
both_error_density = nx.density(n.subgraph((both_binar_indices)))
any_error_density = nx.density(n.subgraph((any_error_indices)))
b_error_density = nx.density(n.subgraph((unique_3b_binar_indices)))
c_error_density = nx.density(n.subgraph((unique_3c_binar_indices)))
b_full_density = nx.density(n.subgraph((b_indices)))
c_full_density = nx.density(n.subgraph((c_indices)))
gold_full_density = nx.density(n.subgraph((neither_binar_indices)))

shared_density
any_error_density
b_error_density
c_error_density
both_error_density
b_full_density
c_full_density
gold_full_density

b_full_density/gold_full_density
c_full_density/gold_full_density

avgDegree(n.subgraph((b_indices)))
avgDegree(n.subgraph((c_indices)))
avgDegree(n.subgraph((neither_binar_indices)))

from networkx.algorithms import community
from networkx.algorithms.community.quality import modularity

sampled_b_indices = random.sample(list(b_indices), 3000)
b_mod_com = community.greedy_modularity_communities(n.subgraph(sampled_b_indices))
modularity(n.subgraph((sampled_b_indices)), b_mod_com)

sampled_c_indices = random.sample(list(c_indices), 3000)
c_mod_com = community.greedy_modularity_communities(n.subgraph(sampled_c_indices))
modularity(n.subgraph((sampled_c_indices)), c_mod_com)

sampled_gold_indices = random.sample(list(neither_binar_indices), 3000)
sampled_gold_com = community.greedy_modularity_communities(n.subgraph(sampled_gold_indices))
modularity(n.subgraph((sampled_gold_indices)), sampled_gold_com)


def avgDegree(graph):
    return np.mean([x[1] for x in n.subgraph((graph)).degree])

len(shared_binar_indices)
len(both_binar_indices)
len(neither_binar_indices)

nx.attribute_assortativity_coefficient(n, 'all_color')

def getAverageEstimates(graph, indices, trials, sample_size, mod_trials, modularity_sample_size):
    out_dict = {
        "modularity": [],
        "avgDegree": [],
        "density": [],
    }
    for i in range(trials):
        sampled_indices = random.sample(list(indices), sample_size)
        out_dict["avgDegree"].append(avgDegree(graph.subgraph((sampled_indices))))
        out_dict["density"].append(nx.density(graph.subgraph((sampled_indices))))
    for i in range(mod_trials):
        sampled_indices = random.sample(list(indices), modularity_sample_size)
        mod_com = community.greedy_modularity_communities(graph.subgraph(sampled_indices))
        out_dict["modularity"].append(modularity(n.subgraph((sampled_indices)), mod_com))

    summary = []
    for x, j in out_dict.items():
        summary.append((x, np.mean(j)))

    return summary

0.00276565723916949/0.0009045840973495583
0.003151854938786094/0.0009045840973495583
getAverageEstimates(n, neither_binar_indices, 15, 6000, 3, 4000)
getAverageEstimates(n, b_indices, 15, 6000, 3, 4000)
getAverageEstimates(n, c_indices, 15, 6000, 3, 4000)
