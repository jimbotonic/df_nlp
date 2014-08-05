# 
# DF is free software; you can redistribute it and/or
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
# License along with DF; see the file COPYING.  If not
# see <http://www.gnu.org/licenses/>.
# 
# Copyright (C) 2014 Jimmy Dubuisson <jimmy.dubuisson@gmail.com>
#

from __future__ import division
from igraph import Graph
from numpy import dot
import re
import random
import yaml
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import cPickle as pickle
from blist import sortedset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from numpy import array
import sys
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import *

from lemmatizer import *
from utils import *

def get_vectors(tm, tf, p_ppr, conf, center=None, g=None, dsub_ids=None, kmats_names=None, dstats=None):
	if not conf['bow']['use_bow']:
		if conf['fp']['use_fp']:
			dm = p_ppr[tm]
			df = p_ppr[tf]
			if not conf['vv']['use_var_vector']:
				vm = MatrixUtils.build_fp_vector(dm,conf['fp']['times'],conf['dim']['reduce_dimension'],center)
				vf = MatrixUtils.build_fp_vector(df,conf['fp']['times'],conf['dim']['reduce_dimension'],center)
			else:
				vm = MatrixUtils.build_fp_var_vector(dm,conf['fp']['times'],conf['dim']['reduce_dimension'],center)
				vf = MatrixUtils.build_fp_var_vector(df,conf['fp']['times'],conf['dim']['reduce_dimension'],center)
		elif conf['cs']['use_clusters']:
			vm = MatrixUtils.build_cluster_vector(g,p_ppr[tm],dsub_ids)
			vf = MatrixUtils.build_cluster_vector(g,p_ppr[tf],dsub_ids)
		else:
			if conf['dim']['reduce_dimension']:
				vm = p_ppr[tm]
				vf = p_ppr[tf]
				vm = MatrixUtils.get_reduced_vector(vm, center)
				vf = MatrixUtils.get_reduced_vector(vf, center)
			else:
				vm = p_ppr[tm]
				vf = p_ppr[tf]
	else:
		if not conf['bow']['bow_weights']:
			vm = MatrixUtils.build_bow_vector(tm,g,kmats_names)
			vf = MatrixUtils.build_bow_vector(tf,g,kmats_names)
		else:
			vm = MatrixUtils.build_bow_vector(tm,g,kmats_names,dstats)
			vf = MatrixUtils.build_bow_vector(tf,g,kmats_names,dstats)
		if conf['dim']['reduce_dimension']:
			vm = MatrixUtils.get_reduced_vector(vm, center)
			vf = MatrixUtils.get_reduced_vector(vf, center)
	return vm,vf

if __name__ == '__main__':
	""" arguments: <g_filename> <ppr_filename> """
	g = pickle.load(open(sys.argv[1],'rb'))
	p_ppr = pickle.load(open(sys.argv[2], 'rb'))

	f = file('config_binary.yml', 'r')
	conf = yaml.load(f)
	f.close()

	# global
	sample_size = conf['global']['sample_size']
	dimension = conf['dim']['dimension']
	
	# standardize? -> seems to improve accuracy of adaboost
	standardize = conf['classification']['standardize']
	# normalize?
	normalize = conf['classification']['normalize']

	kmats_names = None
	dstats = None
	if conf['bow']['use_bow']:
		kmat_folder = sys.argv[3]	
		print 'loading kmats'
		kmats_names = FileUtils.load_items_names(kmat_folder, p_ppr.keys())
		if conf['bow']['bow_weights']:
			doc_folder = sys.argv[4]	
			print 'loading dstats'
			dstats = FileUtils.load_pickle_files(doc_folder, p_ppr.keys())
	
	print 'basic stats for g:'
	GraphUtils.display_graph_stats(g,verbose=conf['global']['display_graph_stats'])
	core,core_vids,shell_vids = GraphUtils.get_core(g)	
	
	print 'basic stats for core:' 
	GraphUtils.display_graph_stats(core,verbose=conf['global']['display_graph_stats'])

	X = []
	y = []

	print 'preparing training set'

	rx = re.compile('.*\.male\..*')
	ppr_keys = p_ppr.keys()
	male_keys = [m for m in ppr_keys if rx.match(m)]
	female_keys = [f for f in ppr_keys if f not in male_keys]
	training_male_keys = random.sample(male_keys,sample_size)
	training_female_keys = random.sample(female_keys,sample_size)
	test_male_keys = [m for m in ppr_keys if m in male_keys and m not in training_male_keys] 
	test_female_keys = [f for f in ppr_keys if f not in male_keys and f not in training_female_keys] 

	# compute average vector
	print 'computing training vectors statistics (min, max, average and variance)'
	dv = {}
	for i in range(sample_size):
		tm = training_male_keys[i]
		tf = training_female_keys[i]
		dv[tm] = p_ppr[tm]
		dv[tf] = p_ppr[tf]
		
	# compute mean/variance/... vector
	v_min,v_max,v_avg,v_var = MatrixUtils.get_min_max_avg_var_vectors(dv)

	# get avg vector for training set (male blogs and female blogs)
	dv1 = {}
	for i in range(sample_size):
		tm = training_male_keys[i]
		dv1[tm] = p_ppr[tm]
	dv2 = {}
	for i in range(sample_size):
		tf = training_female_keys[i]
		dv2[tf] = p_ppr[tf]
	
	v1_min,v1_max,v1_avg,v1_var = MatrixUtils.get_min_max_avg_var_vectors(dv1)
	v2_min,v2_max,v2_avg,v2_var = MatrixUtils.get_min_max_avg_var_vectors(dv2)
	
	# get graph clusters
	dsub_ids = None
	if conf['cs']['use_clusters']:
		print 'clustering the domain graph'
		#clusters = get_graph_clusters(g,5,True,'g_clusters_walktrap_5.p')
		clusters = pickle.load(open('g_clusters_walktrap_4.p','rb'))
		subs = clusters.as_clustering().subgraphs()
		print '# clusters: ', len(subs)
		dsub_ids = MatrixUtils.load_subgraph_vids(g,subs)

	print 'computing pagerank'
	pr = GraphUtils.my_pagerank(g,ignore_weights=conf['global']['ignore_edge_weights'])
	#pickle.dump(pr, open(sys.argv[1].replace('g_', 'pr_'), 'wb'), pickle.HIGHEST_PROTOCOL)
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
			#center = MatrixUtils.get_highest_entry_indices(v_avg, dimension, shell_vids)
			#center = MatrixUtils.get_highest_entry_indices(v_var, dimension, shell_vids)
		else:
			rids = random.sample(range(len(core_vids)), dimension) 
			center = [core_vids[i] for i in rids]

	if conf['dim']['reduce_dimension']:
		v1_avg = MatrixUtils.get_reduced_vector(v1_avg, center)
		v2_avg = MatrixUtils.get_reduced_vector(v2_avg, center)

	print 'training SVM & RF'

	for i in range(sample_size):
		tm = training_male_keys[i]
		tf = training_female_keys[i]
		vm,vf = get_vectors(tm, tf, p_ppr, conf, center, g, dsub_ids, kmats_names, dstats)
		if standardize:
			vm = MatrixUtils.standardize_vector(vm,v_avg,v_var)		
			vf = MatrixUtils.standardize_vector(vf,v_avg,v_var)		
		if normalize:
			vm = MatrixUtils.normalize_vector(vm,v_min,v_max)		
			vf = MatrixUtils.normalize_vector(vf,v_min,v_max)		
		X.append(vm)
		y.append(0)
		X.append(vf)
		y.append(1)	
	
	# SVM & RF
	rbf = svm.SVC(kernel='rbf', C=10000, gamma=0.1)	
	#rbf = svm.SVC(kernel='rbf', C=10, gamma=10)	
	#rbf = svm.LinearSVC()
	#rbf = linear_model.SGDClassifier()
	#rbf = linear_model.Perceptron(n_iter=4, shuffle=True)
	rbf.fit(X, y)  
	
	# whole vectors:
	#rf = RandomForestClassifier(n_estimators=rf_n_estimators)
	#
	# rf = GaussianNB()
	# rf = MultinomialNB()
	rf = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', max_depth=conf['classification']['dtree_depth'], max_features=None, min_density=None, min_samples_leaf=1, min_samples_split=2), algorithm="SAMME", n_estimators=conf['classification']['ada_n_estimators'])
	rf.fit(X, y)  
	
	print 'testing SVM & RF - training data'

	total = sample_size * 2 
	correct1 = 0
	correct2 = 0
	for i in range(sample_size):
		tm = training_male_keys[i]
		tf = training_female_keys[i]
		vm,vf = get_vectors(tm, tf, p_ppr, conf, center, g, dsub_ids, kmats_names, dstats)
		if standardize:
			vm = MatrixUtils.standardize_vector(vm,v_avg,v_var)		
			vf = MatrixUtils.standardize_vector(vf,v_avg,v_var)		
		if normalize:
			vm = MatrixUtils.normalize_vector(vm,v_min,v_max)		
			vf = MatrixUtils.normalize_vector(vf,v_min,v_max)		
		cm1 = rbf.predict([vm])[0]
		cm2 = rf.predict([vm])[0]
		cf1 = rbf.predict([vf])[0]
		cf2 = rf.predict([vf])[0]

		if cm1 == 0:
			correct1 += 1
		if cm2 == 0:
			correct2 += 1
		if cf1 == 1:
			correct1 += 1
		if cf2 == 1:
			correct2 += 1
			
	print 'SVM: ', correct1, '/', total, '=', (correct1/total)*100, '%'
	print 'RF : ', correct2, '/', total, '=', (correct2/total)*100, '%'
	
	print 'testing SVM & RF - test data'

	correct1 = 0
	correct2 = 0
	correct3 = 0
	for i in range(sample_size):
		tm = test_male_keys[i]
		tf = test_female_keys[i]
		vm,vf = get_vectors(tm, tf, p_ppr, conf, center, g, dsub_ids, kmats_names, dstats)

		d_m1 = cityblock(v1_avg,vm)
		d_m2 = cityblock(v2_avg,vm)
		d_f1 = cityblock(v2_avg,vf)
		d_f2 = cityblock(v1_avg,vf)

		if standardize:
			vm = MatrixUtils.standardize_vector(vm,v_avg,v_var)		
			vf = MatrixUtils.standardize_vector(vf,v_avg,v_var)		
		if normalize:
			vm = MatrixUtils.normalize_vector(vm,v_min,v_max)		
			vf = MatrixUtils.normalize_vector(vf,v_min,v_max)		

		cm1 = rbf.predict([vm])[0]
		cm2 = rf.predict([vm])[0]
		cf1 = rbf.predict([vf])[0]
		cf2 = rf.predict([vf])[0]

		if cm1 == 0:
			correct1 += 1
		if cm2 == 0:
			correct2 += 1
		if d_m1 < d_m2:
			correct3 += 1
		###
		if cf1 == 1:
			correct1 += 1
		if cf2 == 1:
			correct2 += 1
		if d_f1 < d_f2:
			correct3 += 1
	
	print 'SVM: ', correct1, '/', total, '=', (correct1/total)*100, '%'
	print 'RF : ', correct2, '/', total, '=', (correct2/total)*100, '%'
	print 'Centers : ', correct3, '/', total, '=', (correct3/total)*100, '%'

