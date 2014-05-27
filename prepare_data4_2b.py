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

def get_fp(g,stats_dir,p):
	a = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24]
	p_stats = pickle.load(open(stats_dir + '/' + p, 'rb'))
	return GraphUtils.get_fingerprint(g,p_stats,a)

if __name__ == '__main__':
	""" compute the diffusion fingerprints (arguments: <association_data_dir> <stats_data_dir> <g_filename> <ncpus> <ppr_filename>) """
	logging.basicConfig(level=logging.DEBUG,format='%(asctime)s:%(levelname)s:%(message)s')
	ppservers = ()
	if len(sys.argv) > 3:
    		ncpus = int(sys.argv[4])
    		js = pp.Server(ncpus, ppservers=ppservers)
	else:
    		js = pp.Server(ppservers=ppservers)
	data_dir = sys.argv[1]
	stats_dir = sys.argv[2]
	fnames = FileUtils.get_files_list(data_dir)
	logging.info('# of files: ' + str(len(fnames)))
	g = pickle.load(open(sys.argv[3], 'rb'))
	p_ppr = {}
	for i in range(10):
		fns = fnames[100*i:100*(i+1)]
		jobs = [(p, js.submit(get_fp,(g,stats_dir,p,),(GraphUtils,MatrixUtils,),('igraph','cPickle','utils','numpy','numpy.linalg',))) for p in fns]
		for p, j in jobs:
			p_ppr[p.replace('.p','')] = j()
	pickle.dump(p_ppr, open(sys.argv[5], "wb"), pickle.HIGHEST_PROTOCOL)
	js.print_stats()	
