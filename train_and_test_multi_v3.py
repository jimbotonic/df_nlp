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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy
import cPickle as pickle
import yaml

from utils import *

if __name__ == '__main__':
	f = file('config_multi.yml', 'r')
	conf = yaml.load(f)
	f.close()

	# needs training?
	train = True
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

	if train:
		X = []
		Y = []
		classes = []

		for f in fnames:
			bows = pickle.load(open(v_dir1 + '/' + f, 'rb'))
			if len(bows) > 2:
				for b in bows:
					if center:
						b = MatrixUtils.get_reduced_vector(b, center)
					X.append(b)
					Y.append(f.replace('.p',''))
					classes.append(f)
		classes = list(set(classes))	
		print '# training posts: ', len(X)
		print '# classes: ', len(classes)

		clf_params = {'n_estimators': 300, 'n_jobs': -1, 'max_depth': 8}
		clf = OneVsRestClassifier(RandomForestClassifier(**clf_params))
		clf.fit(X, Y)
		pickle.dump(clf, open('clf.p', 'wb'), pickle.HIGHEST_PROTOCOL)
		pickle.dump(classes, open('classes.p', 'wb'), pickle.HIGHEST_PROTOCOL)
	else:
		clf = pickle.load(open('clf.p','rb'))
		classes = pickle.load(open('classes.p','rb'))

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
	pclasses = clf.predict(Xtest)

	total = 0
	count = 0
	for i in range(len(pclasses)):
		if pclasses[i] == Ytest[i]:
			count += 1
		total += 1

	print 'random? ', rp
	print 'dimension: ', dimension
	print 'accuracy: ', count, '/', total, '=', (float(count)/total)*100, '%'
