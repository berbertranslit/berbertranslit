import pandas as pd
from collections import Counter
import os
import sys
import ast

from igraph import Graph
import re

with open("transliterate/output/tri_alignments.txt") as f:
    lines = f.readlines()

import ast
res = ast.literal_eval("['b','c']")

lines[0:20]
line.split()[1]
gold_aligned = []
b_aligned = []
c_aligned = []
for i, line in enumerate(lines):
    if line.startswith("gold"):
        gold_aligned.append(ast.literal_eval(line.split("\t")[1].replace("(", "[").replace(")", "]")))
    if line.startswith("tifn"):
        b_aligned.append(ast.literal_eval(line.split("\t")[1].replace("(", "[").replace(")", "]")))
    if line.startswith("lt2"):
        c_aligned.append(ast.literal_eval(line.split("\t")[1].replace("(", "[").replace(")", "]")))

len(c_aligned)

tri_recalc_matris_leccionis = []
for i, x in enumerate(zip(gold_aligned, b_aligned, c_aligned)):
    tri_alignment = list(map(list, zip(*x)))
    replaced = []
    for j, char in enumerate(tri_alignment):
        gold_char = char[0]
        b_char = char[1]
        c_char = char[2]
        if j == len(tri_alignment)-1:
            gold_char_next = " "
        else:
            gold_char_next = tri_alignment[j+1][0]
        if gold_char == "i" and gold_char_next == " " and b_char == "j":
            b_char = "i"
        if gold_char == "i" and gold_char_next == " " and c_char == "j":
            c_char = "i"
        if gold_char == "u" and gold_char_next == " " and b_char == "w":
            b_char = "u"
        if gold_char == "u" and gold_char_next == " " and c_char == "w":
            c_char = "u"
        replaced.append([gold_char, b_char, c_char])
    tri_recalc_matris_leccionis.append(replaced)

tri_alignment = list(map(list, zip(*tri_recalc_matris_leccionis)))

tri_recalc_matris_leccionis

gold_rml = list()
b_rml = list()
c_rml = list()
for utterance in tri_recalc_matris_leccionis:
    cur_gold = ''
    cur_b = ''
    cur_c = ''
    for z, char_pos in enumerate(utterance):
        for j, char in enumerate(char_pos):
            if j == 0:
                if char not in ["SOS", "EOS", "-", "*", " "]:
                    cur_gold += char
                if char == " " or z == len(utterance)-1:
                    gold_rml.append(cur_gold)
                    cur_gold = ''
            elif j == 1:
                if char not in ["SOS", "EOS", "-", "*", " "]:
                    cur_b += char
                if char == " " or z == len(utterance)-1:
                    b_rml.append(cur_b)
                    cur_b = ''
            elif j == 2:
                if char not in ["SOS", "EOS", "-", "*", " "]:
                    cur_c += char
                if char == " " or z == len(utterance)-1:
                    c_rml.append(cur_c)
                    cur_c = ''
def getStripped(word):
    if len(word) in [0, 1]:
        out = word
    else:
        out = re.sub("[aaieəou](?!\b)", "", word[:-1])
        out = out + word[-1]
    return out

gold_rml = list(map(lambda x: getStripped(x), gold_rml))

len(gold_rml)
len(b_rml)
len(c_rml)

"""
all_gold = []
all_pred_3b = []
all_pred_3c = []
for index, line in enumerate(lines):
    if (index) % 6 == 1:
        split = line.strip("\n").split(":")
        for utt_ind, utterance in enumerate(split):
            utterance = utterance.strip()
            if utt_ind == 0:
                words = utterance.split(" ")
                words = [getStripped(x) for x in words]
                all_gold.extend(words)
            if utt_ind == 1:
                words = utterance.split(" ")
                all_pred_3b.extend(words)
            if utt_ind == 2:
                words = utterance.split(" ")
                all_pred_3c.extend(words)
"""
all_gold_counter = Counter(gold_rml)
all_pred_3b_counter = Counter(b_rml)
all_pred_3c_counter = Counter(c_rml)
"""
# Get counters
all_gold_counter = Counter(all_gold)
all_pred_3b_counter = Counter(all_pred_3b)
all_pred_3c_counter = Counter(all_pred_3c)
"""
shared_counter = set(all_gold_counter & all_pred_3b_counter & all_pred_3c_counter)

shared_3b_counter = set(all_gold_counter & all_pred_3b_counter)
shared_3c_counter = set(all_gold_counter & all_pred_3c_counter)

union_counter = all_gold_counter + all_pred_3b_counter + all_pred_3c_counter

len(all_gold_counter)
len(all_pred_3b_counter)
len(all_pred_3c_counter)
len(union_counter)


# Get phon_inv
phon_inv_size_gold = set(l for l in "".join([x for x in all_gold_counter]))
phon_inv_size_b = set(l for l in "".join([x for x in all_pred_3b_counter]))
phon_inv_size_c = set(l for l in "".join([x for x in all_pred_3c_counter]))
set(phon_inv_size_gold).symmetric_difference(set(phon_inv_size_c))
set(phon_inv_size_b).symmetric_difference(set(phon_inv_size_c))

shared_set = set(shared_counter)
shared_3b_set = set(shared_3b_counter)
shared_3c_set = set(shared_3c_counter)
all_gold_set = set(all_gold_counter)
all_3b_set = set(all_pred_3b_counter)
all_3c_set = set(all_pred_3c_counter)
all_tokens_set = all_gold_set | all_3b_set | all_3c_set

error_3b_set = set()
for word_3b in all_3b_set:
    if word_3b not in all_gold_set:
        error_3b_set.add(word_3b)

error_3c_set = set()
for word_3c in all_pred_3c:
    if word_3c not in all_gold_set:
        error_3c_set.add(word_3c)

any_error_set = error_3b_set | error_3c_set


import numpy as np
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import editdistance
import time

union_rml = set(gold_rml) | set(b_rml) | set(c_rml)

#arr1 = np.array(list(union_counter.keys()))
#arr2 = np.array(list(union_counter.keys()))

arr1 = np.array(list(union_rml))
arr2 = np.array(list(union_rml))


def inSet(x, set):
    return x in set

token_df = pd.DataFrame({"tokens": arr1})
token_df["gold"] = token_df["tokens"].apply(inSet, args=[all_gold_set])
token_df["3b"] = token_df["tokens"].apply(inSet, args=[all_3b_set])
token_df["3c"] = token_df["tokens"].apply(inSet, args=[all_3c_set])
token_df["shared_by_all"] = token_df["gold"] & token_df["3b"] & token_df["3c"]
token_df["3b_error"] = token_df["3b"] & ~token_df["gold"]
token_df["3c_error"] = token_df["3c"] & ~token_df["gold"]
token_df["any_error"] = token_df["3b_error"] | token_df["3c_error"]
token_df["shared_error"] = token_df["3b_error"] & token_df["3c_error"]

def where_true_condition(x): return x > 0

gold_indices = np.where(where_true_condition(token_df["gold"].astype(int)))[0]
b_indices = np.where(where_true_condition(token_df["3b"].astype(int)))[0]
c_indices = np.where(where_true_condition(token_df["3c"].astype(int)))[0]
all_shared_indices = np.where(where_true_condition(token_df["shared_by_all"].astype(int)))[0]
b_error_indices = np.where(where_true_condition(token_df["3b_error"].astype(int)))[0]
c_error_indices = np.where(where_true_condition(token_df["3c_error"].astype(int)))[0]
any_error_indices = np.where(where_true_condition(token_df["any_error"].astype(int)))[0]
b_and_c_error_indices = np.where(where_true_condition(token_df["shared_error"].astype(int)))[0]

def get_indices(df, col):
    return np.array(df[col].loc[lambda x: x==True].index)

get_indices(token_df, "any_error")

def getEditMatrix(x, y):
    return cdist(arr2.reshape(-1, 1), arr1.reshape(-1, 1), lambda x, y: editdistance.distance(x[0], y[0]))

for n in [len(arr1)]:
    arr1 = np.array(list(union_rml)[:n])
    arr2 = np.array(list(union_rml)[:n])
    start = time.time()
    matrix = getEditMatrix(arr1, arr2)
    matrix = np.where(matrix > 1, 0, matrix)
    end = time.time()
    print(n, end-start)

n = nx.convert_matrix.from_numpy_array(np.nan_to_num(matrix))
np.save("adj2.npy", matrix)

adj_matr_pd = pd.DataFrame(matrix)
adj_matr_pd.columns = arr1
adj_matr_pd.set_index(arr1, inplace=True)
adj_matr_pd = adj_matr_pd.fillna(0)

import networkx as nx
import random
n = nx.convert_matrix.from_pandas_adjacency(adj_matr_pd)
n = nx.convert_node_labels_to_integers(n)
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
group_values = {}
"""
#Loop through each community in the network
for community_number, community in enumerate(communities):
    #For each member of the community, add their community number and a distinct color
    rand_num = random.randrange(1, 256)
    for name in community:
        modularity_class[name] = community_number
        modularity_color[name] = Viridis256[rand_num]
        unique_3b_binar_dict[name] = unique_3b_binar[name]
        unique_3c_binar_dict[name] = unique_3c_binar[name]
"""

for index, value in enumerate(token_df["3b"].astype(int)):
    if value == 1:
        color = Viridis8[0]
    elif value == 0:
        color = Viridis8[-1]
    unique_3b_color.setdefault(index, color)

for index, value in enumerate(token_df["3c"].astype(int)):
    if value == 1:
        color = Viridis8[0]
    elif value == 0:
        color = Viridis8[-1]
    unique_3c_color.setdefault(index, color)


for index, (b_value, c_value, s_value)  in enumerate(zip(token_df["3b"].astype(int), token_df["3c"].astype(int), token_df["shared_by_all"].astype(int))):
    if s_value == 1:
        color = Oranges8[6]
        group = "shared_by_all"
    elif b_value == 1 and c_value == 1:
        color = Reds8[0]
        group = "both_error"
    elif b_value == 1 and c_value == 0:
        color = Viridis8[0]
        group = "b_error"
    elif b_value == 0 and c_value == 1:
        color = Viridis8[4]
        group = "c_error"
    else:
        color = Viridis8[-1]
        group = "gold_only"

    all_color.setdefault(index, color)
    group_values.setdefault(index, group)

Viridis8[-1]
set(all_color.values())
all_color[0]
name_dict = dict(enumerate(arr1, start=0))

#nx.set_node_attributes(sampled_graph, name='modularity_class', values=modularity_class)
#nx.set_node_attributes(sampled_graph, name='modularity_color', values=modularity_color)
nx.set_node_attributes(sampled_graph, name='all_color', values=all_color)
nx.set_node_attributes(sampled_graph, name='name', values=name_dict)
nx.set_node_attributes(sampled_graph, name='group_values', values=group_values)

nx.get_node_attributes(sampled_graph, 'all_color')

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
    ("all color", "$color[swatch]:all_color"),
    ("group_values", "@group_values")
]

#Create a plot — set dimensions, toolbar, and title
plot = figure(tooltips = HOVER_TOOLTIPS,
              tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
            x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

nx.get_node_attributes(sampled_graph, "1000")

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

gold_indices = np.where(where_true_condition(token_df["gold"].astype(int)))[0]
b_indices = np.where(where_true_condition(token_df["3b"].astype(int)))[0]
c_indices = np.where(where_true_condition(token_df["3c"].astype(int)))[0]
all_shared_indices = np.where(where_true_condition(token_df["shared_by_all"].astype(int)))[0]
b_error_indices = np.where(where_true_condition(token_df["3b_error"].astype(int)))[0]
c_error_indices = np.where(where_true_condition(token_df["3c_error"].astype(int)))[0]
any_error_indices = np.where(where_true_condition(token_df["any_error"].astype(int)))[0]
b_and_c_error_indices = np.where(where_true_condition(token_df["shared_error"].astype(int)))[0]


gold_density = nx.density(n.subgraph((gold_indices)))
b_density = nx.density(n.subgraph((b_indices)))
c_density = nx.density(n.subgraph((c_indices)))
all_shared_density = nx.density(n.subgraph((all_shared_indices)))
b_error_density = nx.density(n.subgraph((b_error_indices)))
c_error_density = nx.density(n.subgraph((c_error_indices)))
any_error_density = nx.density(n.subgraph((any_error_indices)))
b_and_c_error_density = nx.density(n.subgraph((b_and_c_error_indices)))

len(all_shared_indices)
len(b_and_c_error_indices)

b_density/gold_density
c_density/gold_density

avgDegree(n.subgraph((b_indices)))
avgDegree(n.subgraph((c_indices)))
avgDegree(n.subgraph((gold_indices)))

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
getAverageEstimates(n, gold_indices, 15, 6000, 3, 4000)
getAverageEstimates(n, b_indices, 15, 6000, 3, 4000)
getAverageEstimates(n, c_indices, 15, 6000, 3, 4000)


k = 6000
sampled_nodes = random.sample(n.nodes, k)
sampled_graph = n.subgraph(sampled_nodes)

np.array(list(set(nx.algorithms.centrality.degree_centrality(n).values())))

import matplotlib.pyplot as plt

from scipy.stats.kde import gaussian_kde
from numpy import linspace
cent = np.array(list(nx.algorithms.centrality.degree_centrality(n).values()))
len(cent)

degree_vector = np.array(list(nx.degree(n)))[:,1]
degree_vector
token_df["degree"] = degree_vector
token_df["degree_10quant"] = pd.qcut(token_df['degree'], 10)

token_df[token_df["gold"]]["degree_10quant"].value_counts(normalize=True)

degree_quant_df = pd.DataFrame(
    {"gold":token_df[token_df["gold"]]["degree_10quant"].value_counts(normalize=True),
    "3b":token_df[token_df["3b"]]["degree_10quant"].value_counts(normalize=True),
    "3c":token_df[token_df["3c"]]["degree_10quant"].value_counts(normalize=True),
    "any_error":token_df[token_df["any_error"]]["degree_10quant"].value_counts(normalize=True),
    "3b_error":token_df[token_df["3b_error"]]["degree_10quant"].value_counts(normalize=True),
    "3c_error":token_df[token_df["3c_error"]]["degree_10quant"].value_counts(normalize=True)}
    )
degree_quant_df

token_df["degree_centrality"] = np.exp(cent)/sum(np.exp(cent))
token_df["degree_centrality"].plot.density()

main_component = np.array(list([c for c in sorted(nx.connected_components(n), key=len, reverse=True)][0]))
token_df['main_component'] = False
token_df['secondary_component'] = False
token_df.loc[main_component, 'main_component'] = True
token_df.loc[token_df["main_component"] == False, "secondary_component"] = True
sum(token_df['secondary_component'] & token_df['any_error']) / sum(token_df['secondary_component'])
sum(token_df['main_component'] & token_df['any_error']) / sum(token_df['main_component'])

sum(token_df['main_component'] & token_df['3b_error'] & ~token_df['shared_error']) / sum(token_df['main_component'] & token_df['any_error'] & ~token_df['shared_error'])
sum(token_df['main_component'] & token_df['3c_error'] & ~token_df['shared_error']) / sum(token_df['main_component'] & token_df['any_error'] & ~token_df['shared_error'])


sum(token_df['secondary_component'] & token_df['3b_error'] & ~token_df['shared_error']) / sum(token_df['secondary_component'] & token_df['any_error'] & ~token_df['shared_error'])
sum(token_df['secondary_component'] & token_df['3c_error'] & ~token_df['shared_error']) / sum(token_df['secondary_component'] & token_df['any_error'] & ~token_df['shared_error'])


# Get % of gold in main component
sum(token_df['main_component'] & token_df['gold']) / sum(token_df['gold'])

# Get % of 3b in main component
sum(token_df['main_component'] & token_df['3b']) / sum(token_df['3b'])

# Get % of 3c in main component
sum(token_df['main_component'] & token_df['3c']) / sum(token_df['3c'])

# Get % of 3b_error in secondary component and main component
sum(token_df['secondary_component'] & token_df['3b_error']) / sum(token_df['3b_error'])
sum(token_df['main_component'] & token_df['3b_error']) / sum(token_df['3b_error'])

# Get % of 3c_error in secondary component and main component
sum(token_df['secondary_component'] & token_df['3c_error']) / sum(token_df['3c_error'])
sum(token_df['main_component'] & token_df['3c_error']) / sum(token_df['3c_error'])
