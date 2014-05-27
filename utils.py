#
# This file is part of DF.
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
# License along with DF; see the file COPYING.  If not
# see <http://www.gnu.org/licenses/>.
# 
# Copyright (C) 2014 Jimmy Dubuisson <jimmy.dubuisson@gmail.com>
#

from __future__ import division
from igraph import Graph
from numpy import linalg 
from blist import sortedset,sortedlist
from math import log

from os import listdir
from os.path import isfile, join
import re
import shutil
import random

import numpy
from numpy import sqrt
from numpy import var
import itertools
import cPickle as pickle

class AssociationMatrix:
	""" square matrix of association values """
	def __init__(self,vmat,cmat,names):
		self.vmat = vmat
		self.cmat = cmat
		# sorted set of names
		self.names = names
		self.dim = len(names)

class DomainMatrix:
	""" union of the association matrices """
	def __init__(self,vmat,names):
		self.vmat = vmat
		# sorted set of names
		self.names = names
		self.dim = len(names)

	def get_coordinates(self,t1,t2):
		return self.names.index(t1), self.names.index(t2)

class MatrixUtils:
	@staticmethod
	def entropy(v,base=2):
		s = 0.
		for i in v:
			if i != 0:
				s += i*log(i,base)
		return -s
	
	@staticmethod
	def mean(v):
		avg = 0.
		for i in v:
			avg += i
		return avg/len(v)

	@staticmethod
	def variance(v):
		s = 0.
		avg = MatrixUtils.mean(v)
		for i in v:
			s += (i-avg)**2
		return s/len(v)

	@staticmethod
	def standardize_vector(v,v_avg,v_var):
		""" standardized each coordinate of the specified vector """
		a = [(i-j) for i,j in zip(v,v_avg)]
		for i in range(len(v)):
			if v_var[i] != 0:
				a[i] = a[i]/v_var[i]**0.5
			else:
				a[i] = 0
		return a
		
	@staticmethod
	def normalize_vector(v,v_min,v_max):
		""" normalize each coordinate of the specified vector """
		a = [(i-j) for i,j in zip(v,v_min)]
		for i in range(len(v)):
			d = v_max[i] - v_min[i]
			if d != 0:
				a[i] = a[i]/d
			else:
				a[i] = 0
		return a

	@staticmethod
	def get_avg_fp_variance_entropy(d):
		v = 0.
		h = 0.
		for k in d.keys():
			v += MatrixUtils.variance(d[k])
			h += MatrixUtils.entropy(d[k])
		return v/len(d.keys()), h/len(d.keys())
		
	@staticmethod
	def get_min_max_avg_var_vectors(d):
		""" get average and variance of provided set of vectors """
		v_min = []
		v_max = []
		v_avg = []
		v_var = []
		for k in d.keys():
		 	if len(v_avg) == 0:
                        	v_avg = d[k]
                	else:
                        	v_avg = map(sum, zip(v_avg,d[k]))
		v_min = [0]*len(v_avg)
		v_max = [0]*len(v_avg)
		for k in d.keys():
				v_min = [min(i,j) for i,j in zip(d[k],v_min)]
				v_max = [max(i,j) for i,j in zip(d[k],v_max)]
		v_var = [0]*len(v_avg)
		for k in d.keys():
				v_diff = [(i-j)**2 for i,j in zip(d[k],v_avg)]
				v_var = map(sum, zip(v_var,v_diff))
		v_avg = [float(i)/len(d.keys()) for i in v_avg]
		v_var = [float(i)/len(d.keys()) for i in v_var]
		return v_min,v_max,v_avg,v_var
	
	@staticmethod
	def load_subgraph_vids(g,subs):
		""" load the specified subgraph vertex ids in a dictionary """
		dsub_ids = {}
        	c = 0
        	for s in subs:
                	sids = GraphUtils.get_vs(g,s.vs['name']).indices
                	dsub_ids[c] = sids
                	c += 1
		return dsub_ids

	@staticmethod
	def build_cluster_vector(g,ppr,dsub_ids,random_select=True):
		""" build cluster vector """
		v = []
		for s in dsub_ids.keys():
			sids = dsub_ids[s]
			if not random_select:
				a = 0
				for i in sids:	
					a += ppr[i]
				v.append(a/len(sids))
			else:
				i = random.sample(sids,1)[0]
				v.append(ppr[i])
		return v

	@staticmethod
	def build_bow_vector(tm,g,kmats_names,dstats=None):
		""" build bag of words vector """
		lg = len(g.vs)
		# get names associated to document
		mnames = kmats_names[tm]
		# get associated vertex sequence in domain graph
		mvs = GraphUtils.get_vs(g,mnames)
		if not dstats:
			vm = MatrixUtils.get_bow_vector(lg,mvs)
		else:
			# get token stats
			mdstats = dstats[tm]
			vm = MatrixUtils.get_bow_vector(lg,mvs,mdstats)
		return vm

	@staticmethod
	def build_fp_vector(d,times,reduce_dimension=False,ids=None):
		""" build the fingerprint vector from dictionary d """
		v = []
		for t in times:
			if reduce_dimension:
				v.append(MatrixUtils.get_reduced_vector(d[t],ids))
			else:
				v.append(d[t])
		return list(itertools.chain(*v))
	
	@staticmethod
	def build_fp_var_vector(d,times,reduce_dimension=False,ids=None):
		""" build the fingerprint variance vector from dictionary d """
		v = []
		dim = len(d.values()[0])
		for i in range(dim):
			a = []
			for t in times:
				a.append(d[t][i])
			v.append(var(a))
		if reduce_dimension:
			v = MatrixUtils.get_reduced_vector(v,ids)
		return v

	@staticmethod
	def get_bow_vector(l,vs,weights=None):
		v = [0] * l
		vids = vs.indices
		counter = 0
		for i in range(l):
			if counter < len(vids) and i == vids[counter]:
				if not weights:
					# set vector index to 1
					v[i] = 1
				else:
					# set vector index to the frequency of the corresponding token
					vn = vs[counter]['name']
					v[i] = weights[vn].freq
				counter += 1
		return v

        @staticmethod
	def get_reduced_vector(x, indices):
		""" select the specified indices in vector x """
		return [x[i] for i in indices]
        
	@staticmethod
	def get_highest_entry_indices(a,k,ignore_ids=None):
		""" get the k highest entry indices (0-based) in the provided array """
		b = list(a)
		a2 = list(a)
		i = len(b)
		if ignore_ids:
			for i in ignore_ids:
				b[i] = -1
		b.sort()
		indices = []
		count = 0
		while i > 0 and count < k:
			m = b[i-1]
			mi = a2.index(m)
			a2[mi] = -1
			indices.append(mi)
			count += 1
			i -= 1
		return indices

        @staticmethod
        def row_normalize(mat):
                #for i in range(len(mat)): # len is ambiguous for sparse ...
                for i in xrange(mat.shape[0]):
                        s = mat[i,:].sum()
                        if s > 0:
                                mat[i,:] = mat[i,:]/s
                return mat

        @staticmethod
        def zero_low_entries(mat, threshold):
                for i in range(len(mat)):
                        for j in range(len(mat)):
                                if mat[i,j] <= threshold:
                                        mat[i,j] = 0
                return mat

        @staticmethod
        def dist(x,y):
		""" euclidean distance between 2 vectors """
		s = 0
		for i in range(len(x)):
			s += (x[i] - y[i])**2
           	return numpy.sqrt(s)

	@staticmethod
	def get_union_association_matrix(umat,kmat):
		for i in range(len(kmat.names)):
			for j in range(len(kmat.names)):
				t1 = kmat.names[i]
				t2 = kmat.names[j]
				if t1 != t2:
					u,v = umat.get_coordinates(t1,t2)		
					umat.vmat[u,v] += kmat.vmat[i,j]
		return umat

class FileUtils:
	@staticmethod
	def load_pickle_files(folder_path,fnames):
		dic = {}
		for n in fnames:
			dic[n] = pickle.load(open(folder_path + '/' + n + '.p', 'rb'))
		return dic
	
	@staticmethod
	def load_items_names(folder_path,fnames):
		dic = {}
		for n in fnames:
			dic[n] = pickle.load(open(folder_path + '/' + n + '.p', 'rb')).names
		return dic

	@staticmethod
	def get_files_list(dir_path):
		return [f for f in listdir(dir_path) if isfile(join(dir_path,f)) ]

	@staticmethod
	def get_filtered_files_list(dir_path,regex):
		p = re.compile(regex)
		return [f for f in listdir(dir_path) if isfile(join(dir_path,f) and p.match(f)) ]
	
	@staticmethod
	def read_text_file(path):
		f = open(path, 'r')
		txt = f.read()
		f.close()
		return txt

	@staticmethod
	def copy_files_sample(src_dir,tgt_dir,n):
		fl = FileUtils.get_files_list(src_dir)
		sl = random.sample(fl,n)
		for p in sl:
			shutil.copy(src_dir+'/'+p,tgt_dir+'/'+p)		

	@staticmethod
        def copy_file(src_dir,tgt_dir,file_name):
        	shutil.copy(src_dir+'/'+file_name,tgt_dir+'/'+file_name)

	@staticmethod
	def copy_filtered_files_sample(src_dir,tgt_dir,regex,n):
		""" copy files whose names match the regex """
		fl = FileUtils.get_filtered_files_list(src_dir,regex)
		sl = random.sample(fl,n)
		for p in sl:
			shutil.copy(src_dir+'/'+p,tgt_dir+'/'+p)		
	
	@staticmethod
	def batch_replace(s,vold,vnew):
		""" replace all occurences of regexs in vold in the string s """
		for i in range(len(vold)):
			s = re.sub(vold[i],vnew[i],s)
		return s

class GraphUtils:
	@staticmethod
	def get_core(g):
		""" extract the core of the graph """
		components = g.components(mode='strong')
        	core = components.giant()
        	core_vids = GraphUtils.get_vs(g, core.vs['name']).indices
        	shell_vids = [i for i in g.vs.indices if i not in core_vids]
		return core,core_vids,shell_vids

	@staticmethod
	def display_graph_stats(g, verbose=False):
		print '#vertices: ', len(g.vs)
		print '#edges: ', len(g.es)
		print 'density: ', g.density()
		if verbose:
			print 'diameter (undirected): ', g.diameter(directed=False) 
			print 'diameter (directed): ', g.diameter(directed=True) 

	@staticmethod
	def get_graph_clusters(g,s,save=False,filename=None):
		clusters = g.community_walktrap(steps=s)
		if save:
        		pickle.dump(clusters, open(filename, 'wb'))
		return clusters

	@staticmethod
	def get_vs(g,names):
		""" get vertex indices by name """
		return g.vs.select(lambda v: v['name'] in names)

	@staticmethod
	def get_graph_from_matrix(assoc_mat,density,ignore_weights=True):
		""" get graph from association matrix """
		mat = assoc_mat.vmat
		l = mat.shape[0]
		g = Graph(directed=True)
        	g.add_vertices(l)
        	g.vs['name'] = assoc_mat.names
		max_edges = int(l*(l-1)*density)
		a,b = mat.nonzero()
		values = sortedlist()
		es = []
		weights = []
		for z in range(len(a)):
			values.add(mat[a[z],b[z]])
		if mat.nnz < max_edges:
			threshold = min(values)
		else:
			threshold = values[len(values)-(max_edges+1)]
                for z in range(len(a)):
			w = mat[a[z],b[z]]
                	if w >= threshold:
                        	es.append((a[z],b[z]))
				weights.append(w)
                g.add_edges(es)
		if not ignore_weights:
			g.es['weight'] = weights
			# this amounts to row-normalize the adjacency matrix
			GraphUtils.normalize_out_weights(g)
		return g

	@staticmethod
	def normalize_out_weights(g):
		for v in g.vs:
			out_nei = g.incident(v.index,mode='OUT')
			ws = 0.0
			for i in out_nei:
				ws += g.es[i]['weight']
			for i in out_nei:
				g.es[i]['weight'] = g.es[i]['weight']/ws

	@staticmethod
	def get_union_by_name(gk,vnames,enames):
		for vn in gk.vs['name']:
			if vn not in vnames:
				vnames.add(vn)
		for e in gk.es:
			en = gk.vs[e.source]['name'] + '-' + gk.vs[e.target]['name']
			#if sn not in vnames or tn not in vnames or en not in enames:
			if en not in enames:
				enames.add(en)
		return vnames, enames

	@staticmethod
	def get_domain_graph(vnames,enames):
		g = Graph(directed=True)
		g.add_vertices(vnames)
		es = []
		for en in enames:
			sn,tn = en.split('-')
			es.append((sn,tn))
		g.add_edges(es)
		return g 		

	@staticmethod
	def get_personalized_pagerank(g,vnames,normalize=True):
		""" get personalized pagerank """
		vids = g.vs.select(name_in=vnames)
		return g.personalized_pagerank(reset_vertices=vids)

	@staticmethod
	def my_pagerank(g,damping=0.85,epsilon=1e-3,ignore_weights=False):
		n = len(g.vs)
		pr = dict.fromkeys([v.index for v in g.vs], 1.0/n)
		pr2 = pr.copy()
		while True:
			for i in range(n):
				nv = 0.0
				in_nei = g.neighbors(i,mode='IN')
				if len(in_nei)>0:
					if ignore_weights:
						for v in in_nei:
							nv += pr[v]/g.degree(v,mode='OUT')
					else:
						for v in in_nei:
							weight = g.es[g.get_eid(v, i)]['weight']
							nv += pr[v] * weight
				pr2[i] = (1-damping)/n + damping*nv
			dist = MatrixUtils.dist(pr2.values(),pr.values())
			pr = pr2.copy()
			if dist <= epsilon:
				break
		return pr.values()

	@staticmethod
	def get_personalized_vector(g,p_stats,uniform_dist=False):
		""" compute the personalized vector and return the associated vertex sequence """
		pr = dict.fromkeys([v.index for v in g.vs], 0.0)
		pvs = g.vs.select(name_in=p_stats.keys())
		if uniform_dist:
			k = len(pvs)
			for i in pvs.indices:
				pr[i] = 1.0/k
		else:
			s = 0.0
			for t in p_stats.keys():
				s += p_stats[t].freq
			for i in pvs.indices:
				vn = g.vs[i]['name']
				f = float(p_stats[vn].freq)
				pr[i] = f/s
		# return personalized vector & personalized vertex sequence
		return pr,pvs
			
	@staticmethod
	def my_personalized_pagerank(g,p_stats,damping=0.85,epsilon=1e-3,ignore_weights=False):
		n = len(g.vs)
		pr, pvs = GraphUtils.get_personalized_vector(g,p_stats)
		k = len(pvs)
		pr2 = pr.copy()
		# personal vector
		pv = pr.copy()
		while True:
			for i in range(n):
				nv = 0.0
				in_nei = g.neighbors(i,mode='IN')
				if len(in_nei)>0:
					if ignore_weights:
						for v in in_nei:
							nv += pr[v]/g.degree(v,mode='OUT')
					else:
						for v in in_nei:
							weight = g.es[g.get_eid(v, i)]['weight']
							nv += pr[v] * weight
				if i in pvs.indices:
					pr2[i] = (1-damping)*pv[i] + damping*nv
				else:
					pr2[i] = damping*nv
			dist = MatrixUtils.dist(pr2.values(),pr.values())
			pr = pr2.copy()
			if dist <= epsilon:
				break
		return pr.values()

	@staticmethod
	def get_fingerprint(g,p_stats,steps,damping=0.85,ignore_weights=False):
		""" get fingerprint at each specified time """
                n = len(g.vs)
		pr, pvs = GraphUtils.get_personalized_vector(g,p_stats)
                pr2 = pr.copy()
		pv = pr.copy()
		ft = {}
                for step in range(1,max(steps)+1):
                        for i in range(n):
                                nv = 0.0
                                in_nei = g.neighbors(i,mode='IN')
                                if len(in_nei)>0:
					if ignore_weights:
						for v in in_nei:
							nv += pr[v]/g.degree(v,mode='OUT')
					else:
						for v in in_nei:
							weight = g.es[g.get_eid(v, i)]['weight']
							nv += pr[v] * weight
                                if i in pvs.indices:
                                        pr2[i] = (1-damping)*pv[i] + damping*nv
                                else:
                                        pr2[i] = damping*nv
                        pr = pr2.copy()
			if step in steps:
				ft[step] = pr.values()
		return ft
