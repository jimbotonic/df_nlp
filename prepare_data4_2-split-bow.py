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

import sys
from utils import *
from igraph import Graph
import cPickle as pickle
import logging

if __name__ == '__main__':
	""" compute the diffusion fingerprints (arguments: <sample_dir> <g_filename> <bow_dir1> <bow_dir2>) """
	logging.basicConfig(level=logging.DEBUG,format='%(asctime)s:%(levelname)s:%(message)s')
	sample_dir = sys.argv[1]
	fnames = FileUtils.get_files_list(sample_dir, '.*\(1\).*')
	logging.info('# of files: ' + str(len(fnames)))
	g = pickle.load(open(sys.argv[2], 'rb'))
	lg = len(g.vs)
	bow_dir1 = sys.argv[3]
	bow_dir2 = sys.argv[4]
	for f in fnames:
		stats1 = pickle.load(open(sample_dir + '/' + f, 'rb'))
		stats2 = pickle.load(open(sample_dir + '/' + f.replace('(1)','(2)'), 'rb'))
		bows1 = []
		bows2 = []
		for s in stats1:
			mvs = GraphUtils.get_vs(g,s.keys())
			vm = MatrixUtils.get_bow_vector(lg,mvs,s)
			bows1.append(vm)
		for s in stats2:
			mvs = GraphUtils.get_vs(g,s.keys())
			vm = MatrixUtils.get_bow_vector(lg,mvs,s)
			bows2.append(vm)
		pickle.dump(bows1, open(bow_dir1 + '/' + f.replace('(1)',''), "wb"), pickle.HIGHEST_PROTOCOL)
		pickle.dump(bows2, open(bow_dir2 + '/' + f.replace('(1)',''), "wb"), pickle.HIGHEST_PROTOCOL)
