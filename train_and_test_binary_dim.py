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

from sklearn.datasets import load_svmlight_file
import sys
from sklearn import svm
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import yaml
import cPickle as pickle
import math
from random import shuffle
from random import sample
from utils import *

def split_array(a, n_parts=1):
	length = len(a)/n_parts
	return [ a[i*length:(i+1)*length] for i in range(n_parts)]

f = file('config_binary.yml', 'r')
conf = yaml.load(f)
f.close()

#X, y = load_svmlight_file(sys.argv[1])

filename = sys.argv[1]
p_ppr = pickle.load(open(filename, 'rb'))

pr_filename = FileUtils.batch_replace(filename,['p_ppr','p_bow'],['pr', 'pr'])
pr = pickle.load(open(pr_filename, 'rb'))
ppr_keys = p_ppr.keys()
n_parts = conf['cross_validation']['n_parts']

# shuffle keys
shuffle(ppr_keys)
split_keys = split_array(ppr_keys,n_parts)

dimensions = [23629,20000,15000,10000,5000,4000,3000,2000,1000,500,250,100]
dim_scores = {}

print n_parts,'-fold cross validation for different dimensions (', conf['classification']['ada_n_estimators'], ' estimators)'
print '########'

for d in dimensions:
	scores = []
	print 'computing ', n_parts, '-fold cross validation for dimension', d
	if not conf['dim']['random_projection']:
		center = MatrixUtils.get_highest_entry_indices(pr, d)
	else:
		center = sample(range(len(pr)), d) 
	for i in range(n_parts):
		test_keys = split_keys[i]
		if i < n_parts:
			train_split = split_keys[:i] + split_keys[i+1:]
		else:
			train_split = split_keys[:-1]
		train_keys = sum(train_split, [])

		#print '# train keys: ', len(train_keys), len(set(train_keys))
		#print '# test keys: ', len(test_keys), len(set(test_keys))
		#print 'intersection: ', len(set(train_keys).intersection(set(test_keys)))

		X = []
		y = []

		print 'adding training vectors'
		for p in train_keys:
			X.append(MatrixUtils.get_reduced_vector(p_ppr[p], center))
			if p.find('.male.') > 0:
				y.append(1)
			else:
				y.append(0)
			
		### classifiersV
		#clf = svm.SVC(kernel='linear')
		#clf = svm.SVC(kernel='rbf', C=10000, gamma=0.1)
		#clf = svm.SVC(kernel='rbf', C=10, gamma=10)    
		clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', max_depth=conf['classification']['dtree_depth'], max_features=None, min_density=None, min_samples_leaf=1, min_samples_split=2), algorithm="SAMME", n_estimators=conf['classification']['ada_n_estimators'])
		
		print 'training classifier'
		clf.fit(X, y)  

		print 'testing'
		counter = 0.
		correct = 0.
		for p in test_keys:
			t = clf.predict(MatrixUtils.get_reduced_vector(p_ppr[p], center))[0]
			if p.find('.male.') > 0:
				if t == 1:
					correct += 1
			else:
				if t == 0:
					correct += 1
			counter += 1
		
		print correct, '/', counter, ' correctly classified vectors'
		print '---'
		scores.append(correct/counter)
	print scores
	n = len(scores)
	mean = sum(scores)/n
	std = math.sqrt(sum((x-mean)**2 for x in scores)/n)
	print "Accuracy: %0.4f (+/- %0.2f)" % (mean, std)
	print '--------'
	dim_scores[d] = mean
	#shuffle(ppr_keys)
	#split_keys = split_array(ppr_keys,n_parts)
print '########'
print dim_scores
dscores = dim_scores.values()
n = len(dscores)
mean = sum(dscores)/n
std = math.sqrt(sum((x-mean)**2 for x in dscores)/n)
print "Accuracy: %0.4f (+/- %0.2f)" % (mean, std)
