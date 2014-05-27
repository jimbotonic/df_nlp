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

from __future__ import division
from utils import *
from lemmatizer import *
from igraph import Graph
from numpy import dot
import cPickle as pickle
import math

if __name__ == '__main__':
        rgx = '\w+'
        #punct = '\',.!?'
        min_length = 3
        # min number of occurences
        min_occ = 3
        # max frequency (between 0 and 1)
        max_freq = 1
	# min number of tokens
	min_size = 100
	# max number of tokens
	max_size = 1000
	# folder path
	data_dir = sys.argv[1]
	pickle_dir1 = sys.argv[2]
	pickle_dir2 = sys.argv[3]
	# collocation metrics instance to be used
        #cmetrics = CollocationMetrics(CollocationMetrics.decreasing_exp,(1,1),CollocationMetrics.do_nothing,())
        cmetrics = CollocationMetrics(CollocationMetrics.decreasing_exp,(1,1),CollocationMetrics.information,())
	# batch replace arrays
	vold = ['</?blog>','</?Blog>','</?post>','<date>.*</date>','nbsp','urlLink']
	vnew = ['','','','','','']
		
	fnames = FileUtils.get_files_list(data_dir)

	counter = 1
	max_count = 2000
	success_count = 0
	for p in fnames:
		if success_count == max_count:
			break
		print counter, '- Tokenizing: ', p 
		counter += 1
		txt = FileUtils.read_text_file(data_dir + '/' + p)
		txt = FileUtils.batch_replace(txt,vold,vnew)
        	doc = DocStats(txt, rgx, min_length, min_occ, max_freq, cmetrics)
		print '# tokens: ', len(doc.token_set) 
        	if len(doc.token_set) >= min_size and len(doc.token_set) <= max_size:
			i = txt.index(' ',int(math.ceil(len(txt)/2)))
			txt1 = txt[:i]
			txt2 = txt[i:] 
        		doc1 = DocStats(txt1, rgx, min_length, min_occ, max_freq, cmetrics)
        		doc2 = DocStats(txt2, rgx, min_length, min_occ, max_freq, cmetrics)
			mat1 = doc1.get_collocation_mat()
			mat2 = doc2.get_collocation_mat()
			#print '# rows: ', mat.dim
			#print '# nnz entries: ', mat.vmat.nnz
			if mat1 and mat2:
				success_count += 1
				pickle.dump(doc1.token_stats, open(pickle_dir1 + '/' + p.replace('.xml','') + "(1).p", "wb"))
				pickle.dump(doc2.token_stats, open(pickle_dir1 + '/' + p.replace('.xml','') + "(2).p", "wb"))
				pickle.dump(mat1, open(pickle_dir2 + '/' + p.replace('.xml','') + "(1).p", "wb"))
				pickle.dump(mat2, open(pickle_dir2 + '/' + p.replace('.xml','') + "(2).p", "wb"))
		print '---'

