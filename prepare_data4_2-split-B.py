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
import pp

def get_pprs(g,stats_dir,p,iw):
	""" get diffusion fingerprint (use association matrices) """
	p_stats = pickle.load(open(stats_dir + '/' + p, 'rb'))
	pprs = []
	for stat in p_stats:
		print '# tokens for ', p, ': ', len(stat.keys())
		pprs.append(GraphUtils.my_personalized_pagerank(g,stat,ignore_weights=iw))
	print '# vectors: ', len(pprs)
	return pprs

if __name__ == '__main__':
	""" compute the diffusion fingerprints (arguments: <sample_dir> <g_filename> <ncpus> <pprs_dir>) """
	logging.basicConfig(level=logging.DEBUG,format='%(asctime)s:%(levelname)s:%(message)s')
	ppservers = ()
    	ncpus = int(sys.argv[3])
    	js = pp.Server(ncpus, ppservers=ppservers)
	sample_dir = sys.argv[1]
	pprs_dir = sys.argv[4]
	iw = True
	fnames = FileUtils.get_files_list(sample_dir,'.*\(1\).*')
	logging.info('# of files: ' + str(len(fnames)))
	g = pickle.load(open(sys.argv[2], 'rb'))
	p_ppr = {}
	for i in range(100):
		fns = fnames[10*i:10*(i+1)]
		jobs = [(p, js.submit(get_pprs,(g,sample_dir,p,iw,),(GraphUtils,MatrixUtils,),('igraph','cPickle','utils','numpy','numpy.linalg',))) for p in fns]
		for p, j in jobs:
			pickle.dump(j(), open(sys.argv[4] + '/' + p.replace('(1)',''), 'wb'), pickle.HIGHEST_PROTOCOL)
	js.print_stats()	
