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
# Copyright (C) 2014-2019 Jimmy Dubuisson <jimmy.dubuisson@gmail.com>
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
	# min # of posts
	min_posts = 32
	# min number of tokens
	min_size = 100
	# max number of tokens
	max_size = 1000
	# min # tokens per post
	min_tokens = 8
	# min # posts per part
	min_posts_per_part = 12
	
	# folder path
	data_dir = sys.argv[1]
	pickle_dir = sys.argv[2]
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
		print counter, '- Tokenizing: ', p, ' (', success_count, ')' 
		counter += 1
		# get the set of posts for the whole document
		posts = FileUtils.get_blog_posts(data_dir + '/' + p)
		n_posts = len(posts)
		print '# posts', n_posts
		txt = ''
		for post in posts:
			txt += ' ' + post
		txt = FileUtils.read_text_file(data_dir + '/' + p)
		txt = FileUtils.batch_replace(txt,vold,vnew)
        	main_doc = DocStats(txt, rgx, min_length, min_occ, max_freq, cmetrics)
		n_tokens = len(main_doc.token_set)
		print '# tokens: ', n_tokens
        	if n_posts >= min_posts and n_tokens >= min_size and n_tokens <= max_size:
			index = int(math.floor(n_posts/2)) + 1
			txt1 = ''
			docs1 = []
			docs2 = []
			for i in range(index):
				txt1 += ' ' + posts[i]
        			doc = DocStats(posts[i], rgx, min_length, min_occ, max_freq, cmetrics)
				if len(doc.token_stats.keys()) >= min_tokens:
					docs1.append(doc.token_stats)
			for i in range(index, n_posts):
        			doc = DocStats(posts[i], rgx, min_length, min_occ, max_freq, cmetrics)
				if len(doc.token_stats.keys()) >= min_tokens:
					docs2.append(doc.token_stats)
			if len(docs1) >= min_posts_per_part and len(docs2) >= min_posts_per_part:
				doc_part1 = DocStats(txt1, rgx, min_length, min_occ, max_freq, cmetrics)
				mat_part1 = doc_part1.get_collocation_mat()
				if mat_part1:
					success_count += 1
					pickle.dump(docs1, open(pickle_dir + '/' + p.replace('.xml','') + "(1).p", "wb"), pickle.HIGHEST_PROTOCOL)
					pickle.dump(main_doc.token_stats, open(pickle_dir + '/' + p.replace('.xml','') + "(main-doc).p", "wb"), pickle.HIGHEST_PROTOCOL)
					pickle.dump(doc_part1.token_stats, open(pickle_dir + '/' + p.replace('.xml','') + "(doc-part1).p", "wb"), pickle.HIGHEST_PROTOCOL)
					pickle.dump(mat_part1, open(pickle_dir + '/' + p.replace('.xml','') + "(mat-part1).p", "wb"), pickle.HIGHEST_PROTOCOL)
					pickle.dump(docs2, open(pickle_dir + '/' + p.replace('.xml','') + "(2).p", "wb"), pickle.HIGHEST_PROTOCOL)
		print '---'

