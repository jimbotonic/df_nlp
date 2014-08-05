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

def get_ppr(g,stats_dir,p,iw):
	""" get diffusion fingerprint (use association matrices) """
	p_stats = pickle.load(open(stats_dir + '/' + p, 'rb'))
	return GraphUtils.my_personalized_pagerank(g,p_stats,ignore_weights=iw)

if __name__ == '__main__':
	""" compute the diffusion fingerprints (arguments: <sample_dir> <g_filename> <ncpus> <ppr_filename>) """
	logging.basicConfig(level=logging.DEBUG,format='%(asctime)s:%(levelname)s:%(message)s')
	ppservers = ()
    	ncpus = int(sys.argv[3])
    	js = pp.Server(ncpus, ppservers=ppservers)
	sample_dir = sys.argv[1]
	iw = True
	fnames = FileUtils.get_files_list(sample_dir,'.*\(doc-part1\).*')
	logging.info('# of files: ' + str(len(fnames)))
	g = pickle.load(open(sys.argv[2], 'rb'))
	p_ppr = {}
	for i in range(10):
		fns = fnames[100*i:100*(i+1)]
		jobs = [(p, js.submit(get_ppr,(g,sample_dir,p,iw,),(GraphUtils,MatrixUtils,),('igraph','cPickle','utils','numpy','numpy.linalg',))) for p in fns]
		for p, j in jobs:
			p_ppr[p.replace('.p','')] = j()
	pickle.dump(p_ppr, open(sys.argv[4], "wb"), pickle.HIGHEST_PROTOCOL)
	js.print_stats()	
