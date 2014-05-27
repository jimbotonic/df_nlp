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
import sys
from utils import *
from igraph import Graph
import cPickle as pickle
import logging as log

if __name__ == '__main__':
	log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')
	data_dir = sys.argv[1]
	graph_file_name = sys.argv[2]
	fnames = FileUtils.get_files_list(data_dir)
	vnames = sortedset()
	enames = sortedset()
	counter = 1
	for p in fnames:
		log.info(str(counter) + '- Adding graph: ' + p) 
		counter += 1
		gk = pickle.load(open(data_dir + '/' + p, 'rb'))
		vnames, enames = GraphUtils.get_union_by_name(gk,vnames, enames)
		log.info('# vertices/# edges/density: ' + str(len(vnames)) + ' ' + str(len(enames)) + ' ' + str(len(enames)/(len(vnames)*(len(vnames)-1))))
		if counter == 1000:
			break

	g = GraphUtils.get_domain_graph(vnames, enames)
	pickle.dump(g, open(graph_file_name, 'wb'))
	
	
