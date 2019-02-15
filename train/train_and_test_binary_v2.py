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

def split_array(a, n_parts=1):
	length = len(a)/n_parts
	return [ a[i*length:(i+1)*length] for i in range(n_parts)]

f = file('config_binary.yml', 'r')
conf = yaml.load(f)
f.close()

#X, y = load_svmlight_file(sys.argv[1])

p_ppr = pickle.load(open(sys.argv[1], 'rb'))
ppr_keys = p_ppr.keys()
n_parts = conf['cross_validation']['n_parts']
split_keys = split_array(ppr_keys,n_parts)
	
### cross validation 1
#scores = cross_validation.cross_val_score(clf, X.toarray(), y, cv=10)

rscores = []

print 'repeated ', n_parts,'-fold cross validation for the full dimensional vectors (', conf['classification']['ada_n_estimators'], ' estimators)'
print '########'

for k in range(10):
	scores = []
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
			X.append(p_ppr[p])
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
			t = clf.predict(p_ppr[p])[0]
			if p.find('.male.') > 0:
				if t == 1:
					correct += 1
			else:
				if t == 0:
					correct += 1
			counter += 1
		print correct, '/', counter, ' correctly classified vectors'
		scores.append(correct/counter)
		print '---'
	print scores
	n = len(scores)
	mean = sum(scores)/n
	std = math.sqrt(sum((x-mean)**2 for x in scores)/n)
	print "Accuracy: %0.4f (+/- %0.2f)" % (mean, std)
	print '--------'
	print 'shuffling data'
	rscores.append(mean)
	shuffle(ppr_keys)
	split_keys = split_array(ppr_keys,n_parts)
print '########'
print rscores
n = len(rscores)
mean = sum(rscores)/n
std = math.sqrt(sum((x-mean)**2 for x in rscores)/n)
print "Accuracy: %0.4f (+/- %0.2f)" % (mean, std)
