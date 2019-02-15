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
# Copyright (C) 2014 Jimmy Dubuisson <jimmy.dubuisson@gmail.com>
#

import sys
from utils import *
from igraph import Graph
import cPickle as pickle
import logging
import pp

def get_ppr(g,data_dir,p):
	""" get diffusion fingerrprint (use association graphs) """
	gk = pickle.load(open(data_dir + '/' + p, 'rb'))
	return GraphUtils.my_personalized_pagerank(g,gk.vs['name'], normalize=True)

if __name__ == '__main__':
	""" compute the diffusion fingerprints (arguments: <data_dir> <g_filename> <ncpus> <ppr_filename>) """
	logging.basicConfig(level=logging.DEBUG,format='%(asctime)s:%(levelname)s:%(message)s')
	ppservers = ()
	if len(sys.argv) > 3:
    		ncpus = int(sys.argv[3])
    		js = pp.Server(ncpus, ppservers=ppservers)
	else:
    		js = pp.Server(ppservers=ppservers)
	data_dir = sys.argv[1]
	fnames = FileUtils.get_files_list(data_dir)
	logging.info('# of files: ' + str(len(fnames)))
	g = pickle.load(open(sys.argv[2], 'rb'))
	p_ppr = {}
	for i in range(10):
		fns = fnames[100*i:100*(i+1)]
		jobs = [(p, js.submit(get_ppr,(g,data_dir,p,),(GraphUtils,MatrixUtils,),('igraph','cPickle','utils','numpy','numpy.linalg',))) for p in fns]
		for p, j in jobs:
			p_ppr[p.replace('.p','')] = j()
	pickle.dump(p_ppr, open(sys.argv[4], "wb"), pickle.HIGHEST_PROTOCOL)
	js.print_stats()	
