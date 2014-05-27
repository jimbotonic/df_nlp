# 
# Latassan is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation;
# either version 3 of the License, or (at your option) any
# later version.
#
# Latassan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public
# License along with Latassan; see the file COPYING.  If not
# see <http://www.gnu.org/licenses/>.
# 
# Copyright (C) 2014 Jimmy Dubuisson <jimmy.dubuisson@gmail.com>
#

from __future__ import division
from igraph import Graph
from numpy import dot
import cPickle as pickle
import re
import random
from numpy import array
import random
import sys
from scipy.spatial.distance import *
from scipy.stats import pearsonr, spearmanr
import yaml

from lemmatizer import *
from utils import *

def get_vector(p, p_ppr, conf, vectors, center=None, g=None, kmats_names=None, dstats=None): 
	if vectors.has_key(p):
		return vectors[p]
	if not conf['bow']['use_bow']:
		if conf['fp']['use_fp']:
			d = p_ppr[p]
			if not conf['vv']['use_var_vector']:
				v = MatrixUtils.build_fp_vector(d,conf['fp']['times'],conf['dim']['reduce_dimension'],center)
			else:
				v = MatrixUtils.build_fp_var_vector(d,conf['fp']['times'],conf['dim']['reduce_dimension'],center)
		else:
			if conf['dim']['reduce_dimension']:
				v = MatrixUtils.get_reduced_vector(p_ppr[p], center)
			else:
				v = p_ppr[p]
	else:
		if not conf['bow']['bow_weights']:
			v = MatrixUtils.build_bow_vector(p,g,kmats_names)
		else:
			v = MatrixUtils.build_bow_vector(p,g,kmats_names,dstats)
		if conf['dim']['reduce_dimension']:
			v = MatrixUtils.get_reduced_vector(v, center)
	vectors[p] = v
	return v

if __name__ == '__main__':
	""" arguments: <g_filename> <ppr_filename1> <ppr_filename2> """
	# RUN 1 (L2 norm) -> cannot be reproduced!?
	# full: 4.1%, 1000: 8.7%, 200: 14.3%, 160: 15.5%, 150: 15.5%, 140: 15.5%, 130: 14.9%, 125: 16%, 120: 15.8%, 100: 14.8%
	
	# RUN 2 (L1 norm)
	# #vertices, #edges, density:  15168 1150266 0.00500000312971
	# giant component #vertices, #edges:  15149 1150118 0.00501190855459
	# full: 0.2%, 125: 15.2%, 122: 15.2%, 120: 15.4%, 119: 15.7%, 118:16%, 117: 16.4%, 116: 15.8%, 115: 15.3%
	f = file('config_multi.yml', 'r')
	conf = yaml.load(f)
	f.close()

	# dimension
	dimension = conf['dim']['dimension']
	
	# standardize? -> seems to improve accuracy of adaboost
	standardize = conf['classification']['standardize']
	# normalize?
	normalize = conf['classification']['normalize']

	vectors = {}

	g = pickle.load(open(sys.argv[1],'rb'))
	p_ppr1 = pickle.load(open(sys.argv[2], 'rb'))
	p_ppr2 = pickle.load(open(sys.argv[3], 'rb'))
	
	# compute average vector
	print 'computing vectors min, max, average and variance'
	dv = {}
	for k in p_ppr1.keys():
		dv[k] = p_ppr1[k]
	for k in p_ppr2.keys():
		dv[k] = p_ppr2[k]
		
	# compute mean/variance/... vector
	v_min,v_max,v_avg,v_var = MatrixUtils.get_min_max_avg_var_vectors(dv)

	print 'basic stats for g:'
	GraphUtils.display_graph_stats(g,verbose=conf['global']['display_graph_stats'])
	core,core_vids,shell_vids = GraphUtils.get_core(g)	
	
	print 'basic stats for core:' 
	GraphUtils.display_graph_stats(core,verbose=conf['global']['display_graph_stats'])

	print 'computing pagerank'
	pr = GraphUtils.my_pagerank(g,ignore_weights=conf['global']['ignore_edge_weights'])
	print 'Pagerank entropy: ', MatrixUtils.entropy(pr)
	print 'Pagerank variance: ', MatrixUtils.variance(pr)
	print 'PPR avg FP coordinate variance: ', MatrixUtils.mean(v_var)
	v,h = MatrixUtils.get_avg_fp_variance_entropy(dv)
	print 'PPR avg FP vector variance/entropy: ', v,'/', h

	print 'correlation pagerank <-> variance'
	print spearmanr(pr,v_var)
	print pearsonr(pr,v_var)
	
	print 'correlation pagerank <-> avg'
	print spearmanr(pr,v_avg)
	print pearsonr(pr,v_avg)
	
	print 'correlation variance <-> avg'
	print spearmanr(v_var,v_avg)
	print pearsonr(v_var,v_avg)

	# set coordinates corresponding to shell vertices to -1
	pr = map(lambda (i,x): -1 if i in shell_vids else x, enumerate(pr))

	print 'computing center for reduced dimension'
	center = None
	if conf['dim']['reduce_dimension']:	
		if not conf['dim']['random_projection']:
			center = MatrixUtils.get_highest_entry_indices(pr, dimension, shell_vids)
		else:
			rids = random.sample(range(len(core_vids)), dimension) 
			center = [core_vids[i] for i in rids]

	kmats_names1 = None
	kmats_names2 = None
	dstats1 = None
	dstats2 = None

	if conf['bow']['use_bow']:
		print 'loading kmats 1'
		kmat_folder = sys.argv[4]	
		kmats_names1 = FileUtils.load_items_names(kmat_folder, p_ppr1.keys())
		print 'loading kmats 2'
		kmats_names2 = FileUtils.load_items_names(kmat_folder, p_ppr2.keys())
		if conf['bow']['bow_weights']:
			print 'loading dstats 1'
			doc_folder = sys.argv[5]	
			dstats1 = FileUtils.load_pickle_files(doc_folder,p_ppr1.keys())
			print 'loading dstats 2'
			dstats2 = FileUtils.load_pickle_files(doc_folder,p_ppr2.keys())

	print 'computing vector distances'
	
	total = 0
	correct = 0

	for p2 in p_ppr2.keys():
		total += 1
		print 'comparing ', p2, total
		maxv = 0
		minv = 1
		tv = 0
		pmin = ''
		v2 = get_vector(p2, p_ppr2, conf, vectors, center, g, kmats_names2, dstats2)
		if standardize:
			v2 = MatrixUtils.standardize_vector(v2,v_avg,v_var)		
		if normalize:
			v2 = MatrixUtils.normalize_vector(v2,v_min,v_max)		
		for p1 in p_ppr1.keys():
			v1 = get_vector(p1, p_ppr1, conf, vectors, center, g, kmats_names1, dstats1)
			if standardize:
				v1 = MatrixUtils.standardize_vector(v1,v_avg,v_var)		
			if normalize:
				v1 = MatrixUtils.normalize_vector(v1,v_min,v_max)		
			# http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
			# cityblock: 15.2%, euclidean: 9%, cosine: 8.1%, jaccard. 0.2%, chebyshev: 2.2%i, canberra: 0.2%
			#v = euclidean(v1,v2)
			v = cityblock(v1,v2)
			#v = jaccard(v1,v2)
			#v = chebyshev(v1,v2)
			#v = cosine(v1,v2)
			#v = canberra(v1,v2)
			if v > maxv:
				maxv = v
			if v < minv:
				minv = v
				pmin = p1
			if p1.replace('(1)','(2)') == p2:
				tv = v
				#if tv == minv:
					#pmin = p1
		if pmin.replace('(1)','(2)') == p2:
			correct += 1
		#else:
			#print 'total: ', total, '- true: ', tv, '(max: ', maxv, ' min: ', minv, ')'

	print 'DOT: ', correct, '/', total, '=', (correct/total)*100, '%'
