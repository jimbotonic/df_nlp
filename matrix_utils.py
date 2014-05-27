#
# This file is part of Latassan.
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
import numpy
from numpy import sqrt
from blist import sortedset
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
	def build_bow_vector(tm,g,kmat_folder,doc_folder=None):
		lg = len(g.vs)
		mnames = pickle.load(open(kmat_folder + '/' + tm + '.p', 'rb')).names
		mvs = GraphUtils.get_vs(g,mnames)
		if not doc_folder:
			vm = MatrixUtils.get_bow_vector(lg,mvs)
		else:
			mdstats = pickle.load(open(doc_folder + '/' + tm + '.p', 'rb'))
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
	def get_bow_vector(l,vs,weights=None):
		v = [0] * l
		vids = vs.indices
		counter = 0
		for i in range(l):
			if counter < len(vids) and i == vids[counter]:
				if not weights:
					v[i] = 1
				else:
					vn = vs[counter]['name']
					v[i] = weights[vn].freq
				counter += 1
		return v

        @staticmethod
	def get_reduced_vector(x, indices):
		""" select the specified indices in vector x """
		return [x[i] for i in indices]
        
	@staticmethod
	def get_highest_entry_indices(a,k):
		""" get the k highest entry indices (0-based) in the provided array """
		b = list(a)
		a2 = list(a)
		b.sort()
		i = len(b)
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

