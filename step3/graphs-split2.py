#
# This file is part of DF.
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
# Copyright (C) 2014-2019 Jimmy Dubuisson <jimmy.dubuisson@gmail.com>
#

from __future__ import division
import sys
from utils import *
import re
from igraph import Graph
import cPickle as pickle
import logging as log
import scipy.sparse as sps

if __name__ == '__main__':
	log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')
	data_dir = sys.argv[1]
	graph_file_name = sys.argv[2]
	fnames = FileUtils.get_files_list(data_dir)
	names = sortedset()
	rx = re.compile('.*\(mat-part1\).*')
	density = 0.01
	counter = 1
	
	# first pass: get names
	log.info('getting names...') 
	for p in fnames:
		if rx.match(p):
			log.info(str(counter) + '- Adding words of matrix: ' + p) 
			counter += 1
			kmat = pickle.load(open(data_dir + '/' + p, 'rb'))
			names = names.union(kmat.names)
			if counter == 1000:
				break
	umat = DomainMatrix(sps.lil_matrix((len(names),len(names)),dtype=float), names)
	counter = 1	
	# second pass: adding K matrices
	for p in fnames:
		if rx.match(p):
			log.info(str(counter) + '- Adding matrix: ' + p) 
			counter += 1
			kmat = pickle.load(open(data_dir + '/' + p, 'rb'))
			umat = MatrixUtils.get_union_association_matrix(umat,kmat)
			if counter == 1000:
				break

	pickle.dump(umat, open('domain-mat_' + graph_file_name, 'wb'), pickle.HIGHEST_PROTOCOL)
	g = GraphUtils.get_graph_from_matrix(umat,density)
	log.info('#vertices, #edges: ' + str(len(g.vs)) + ', ' + str(len(g.es))) 
	pickle.dump(g, open(graph_file_name, 'wb'), pickle.HIGHEST_PROTOCOL)
	
	
