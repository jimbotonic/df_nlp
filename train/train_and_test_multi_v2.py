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
import cPickle as pickle
from scipy.spatial.distance import *
import yaml

from utils import *

def is_nearest(ref_p, pr, p_ppr, dist_func, center=None):
	if not center:
		npr=pr
		rpr=p_ppr[ref_p]
	else:
		npr = MatrixUtils.get_reduced_vector(pr, center)
		rpr = MatrixUtils.get_reduced_vector(p_ppr[ref_p], center)
	dmin = dist_func(npr,rpr)
	for k in p_ppr.keys():
		if not center:
			kpr = p_ppr[k]
		else:
			kpr = MatrixUtils.get_reduced_vector(p_ppr[k], center)
		cdist = dist_func(npr,kpr)
		if k != ref_p and cdist <= dmin:
				return False
	return True
		
if __name__ == '__main__':
	f = file('config_multi.yml', 'r')
	conf = yaml.load(f)
	f.close()

	p_ppr = pickle.load(open(sys.argv[1],'rb'))
	pprs_dir = sys.argv[2]
	pprs = FileUtils.get_files_list(pprs_dir)

	center = None
	g = None
	if conf['dim']['reduce_dimension']:	
		dimension = conf['dim']['dimension']
		g = pickle.load(open(sys.argv[3],'rb'))
		print '# vertices: ', len(g.vs), '# edges: ', len(g.es)
		core,core_vids,shell_vids = GraphUtils.get_core(g)	
		pr = GraphUtils.my_pagerank(g,ignore_weights=conf['global']['ignore_edge_weights'])
		pr = map(lambda (i,x): -1 if i in shell_vids else x, enumerate(pr))
		if not conf['dim']['random_projection']:
			center = MatrixUtils.get_highest_entry_indices(pr, dimension, shell_vids)
		else:
			rids = random.sample(range(len(core_vids)), dimension) 
			center = [core_vids[i] for i in rids]

	total = 0
	success = 0
	count = 0
	for p in pprs:
		count += 1
		print 'checking ', p, '(', count, ',', success, '/', total, ')'
		ppra = pickle.load(open(pprs_dir + '/' + p,'rb'))
		for pr in ppra:
			total += 1
			if is_nearest(p.replace('.p','(doc-part1)'), pr, p_ppr, cityblock, center):
				success += 1

	print success, '/', total, '=', (float(success)/total)*100, '%'
