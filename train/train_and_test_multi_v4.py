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
import os.path
import numpy
import cPickle as pickle
import yaml
from scipy.spatial.distance import *

from utils import *

if __name__ == '__main__':
	f = file('config_multi.yml', 'r')
	conf = yaml.load(f)
	f.close()

	# random projection?
	rp = conf['dim']['random_projection']
	# ignore weights?
	ig = conf['global']['ignore_edge_weights'] 

	v_dir1 = sys.argv[1]
	v_dir2 = sys.argv[2]

	center = None
	dimension = None

	if len(sys.argv) == 5:
		g = pickle.load(open(sys.argv[3],'rb'))
		print '# vertices: ', len(g.vs), ' # edges: ', len(g.es)
		dimension = int(sys.argv[4])
		center = GraphUtils.get_center_vids(g,dimension,rp=rp,ig=ig)

	fnames = FileUtils.get_files_list(v_dir1)

	clf = None
	classes = None

	YX = {}
	classes = []

	for f in fnames:
		X = []
		bows = pickle.load(open(v_dir1 + '/' + f, 'rb'))
		if len(bows) > 2:
			for b in bows:
				if center:
					b = MatrixUtils.get_reduced_vector(b, center)
				X.append(b)
			YX[f.replace('.p','')] =  [sum(i)/len(i) for i in zip(*X)]
			classes.append(f)
	print '# training centers: ', len(YX.keys())
	print '# classes: ', len(classes)

	Xtest = []
	Ytest = []
	
	for f in fnames:
		if f in classes and os.path.isfile(v_dir2 + '/' + f):
			bows = pickle.load(open(v_dir2 + '/' + f, 'rb'))
			#print '# BOW vectors: ', len(bows)
			for b in bows:
				if center:
					b = MatrixUtils.get_reduced_vector(b, center)
				Xtest.append(b)
				Ytest.append(f.replace('.p',''))

	print '# posts to test: ', len(Xtest)

	total = 0
	count = 0

	for i in range(len(Xtest)):
		total += 1
		dmin = sys.maxint
		x = Xtest[i]
		pref = Ytest[i]
		pfound = ''
		for p in YX:
			d = cityblock(x,YX[p])
			if d < dmin:
				dmin = d
				pfound = p
		if pfound == pref:
			count += 1

	print 'random? ', rp
	print 'dimension: ', dimension
	print 'accuracy: ', count, '/', total, '=', (float(count)/total)*100, '%'
