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
from igraph import Graph
import cPickle as pickle
import logging as log
import scipy.sparse as sps

if __name__ == '__main__':
	log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')
	matrix_file_name = sys.argv[1]
	graph_file_name = sys.argv[2]
	density = float(sys.argv[3])
	iw = False

	umat = pickle.load(open(matrix_file_name, 'rb'))
	g = GraphUtils.get_graph_from_matrix(umat,density,ignore_weights=iw)
	log.info('#vertices, #edges: ' + str(len(g.vs)) + ', ' + str(len(g.es))) 
	pickle.dump(g, open(graph_file_name, 'wb'))
	
