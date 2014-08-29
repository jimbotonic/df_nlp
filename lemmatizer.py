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

from __future__ import division
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer, RegexpTokenizer, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist
import numpy
from numpy import ceil
import scipy.sparse as sps
import math
import sys
from copy import copy
from igraph import Graph
from utils import *
from blist import sortedset

class TokenUtils:
	### stop words removal
	@staticmethod
	def remove_stopwords(tokens, lang='english'):
		""" remove stopwords from a list of words for the specified language (e.g. 'english') """
		return [w for w in tokens if not w in stopwords.words(lang)]

	@staticmethod
	def remove_shortwords(tokens, l):
		""" remove tokens whose length is lower than specified maximal length """
		return [w for w in tokens if len(w)>=l]

	### tokenization
	#
	# NLTK tokenize module
	# http://nltk.googlecode.com/svn/trunk/doc/api/nltk.tokenize-module.html
	@staticmethod
	def regex_tokenizer(text, regex):
		""" tokenize text according to provided regex (e.g., '\W+') """
		return RegexpTokenizer(regex).tokenize(text)

	@staticmethod
	def word_tokenizer(text):
		return word_tokenize(text)

	@staticmethod
	def wordpunct_tokenizer(text):
		return WordPunctTokenizer().tokenize(text)

	@staticmethod
	def remove_punct_tokens(tokens, l):
		return filter(lambda w: w not in l, tokens)

	### lemmatization
	#
	# NLTK stemmer module
	# http://nltk.googlecode.com/svn/trunk/doc/api/nltk.stem-module.html
	@staticmethod
	def wordnet_lemmatizer(tokens):
		lemmatizer = WordNetLemmatizer()
		return [lemmatizer.lemmatize(w) for w in tokens]

	@staticmethod
	def tokens_to_lowercase(tokens):
		return [w.lower() for w in tokens]

	### word frequencies
	#
	# http://nltk.googlecode.com/svn-/trunk/doc/api/nltk.probability.FreqDist-class.html
	@staticmethod
	def get_token_frequencies(tokens):
		return FreqDist(tokens)

	@staticmethod
	def get_most_freq_tokens(freq,l):
		return dict((k,v) for k, v in freq.iteritems() if v >= l)

	### collocation metrics
	@staticmethod
	def get_token_indices(w,tokens):
    		indices = []
    		offset = -1
    		while True:
        		try:
            			offset = tokens.index(w, offset+1)
        		except ValueError:
            			return indices
        		indices.append(offset)

class CollocationMetrics:
	""" compute collocation metrics between 2 tokens """
	def __init__(self,f,args1,g=None,args2=None):
		self.f = f
		self.args1 = args1
		self.g = g
		self.args2 = args2

	### f functions
	@staticmethod
	def gaussian(i,j,mu=0,sigma=1):
		""" standard normal distribution centered at 0 """
		return ((sigma*math.sqrt(2*math.pi))**(-1))*math.exp(-0.5*(((j-i-1)-mu)/sigma)**2)
	
	@staticmethod
	def decreasing_exp(i,j,alpha=1,beta=1):
		""" decreasing exponential such that f(0)=1 """
		return math.exp(-alpha*(j-i-1)**beta)
	
	@staticmethod
	def decreasing_sigmoid(i,j,offset,alpha=1,beta=1):
		v = math.exp(-alpha*(j-i-1-offset)**beta)
		return v/(1+v)
	
	### g functions
	@staticmethod	
	def mean(value, count, total):
		if value == 0:
			return 0
		else:
			return value/total
	
	@staticmethod
	def information(value, count, total):
		if value == 0:
			return 0
		else:
			return value*(-math.log(count/total))
	
	###

	def distance(self,i,j):
		""" compute distance between indices i and j """
		return self.f(i,j,*self.args1)
	
	def get_collocation_distance(self,indices1,indices2):
		s = 0
		# number of pairs (i,j) such that j>i
		counter = 0
		l1 = len(indices1)
		l2 = len(indices2)
		max_index = indices2[l2-1] + 1
		for i1 in range(l1):
			i = indices1[i1]
			if i1 < l1-1:
				k = indices1[i1+1]
			else:
				k = max_index
			for i2 in range(counter,l2):
				j = indices2[i2]
				if j>i and j < k:
					s = s + self.distance(i,j)
					counter = counter + 1
		return s,counter
			
class TokenStats:
	""" stats for a given token (token, frequency, position indices) """
	def __init__(self,token,freq,indices):
		self.token = token
		self.freq = freq
		self.indices = indices

class DocStats:
	""" stats for a given document (document, regex to separate tokens, minimal length of words, set of tokens, array of token stats) """
	def __init__(self,doc,rgx,min_l,min_o,max_f,metrics):
		self.doc = doc
		# regex expression used to extract tokens
		self.rgx = rgx
		# min token length
		self.min_length = min_l
		# min token # occurences
		self.min_occ = min_o
		# max token frequency
		self.max_freq = max_f
		self.metrics = metrics
		# sequence of tokens
		self.tokens = self.get_tokens()
		# sorted set of tokens
		self.token_set = self.get_sorted_token_set()
		# token_stats: <token> -> <tokenstats instance>
		self.freqs, self.token_stats = self.get_token_stats()

	def get_tokens(self):
		""" extract document tokens """
		doc_tokens = []
		tokens = TokenUtils.regex_tokenizer(self.doc, self.rgx)
		l_tokens =  TokenUtils.tokens_to_lowercase(tokens)
		lst_tokens = TokenUtils.remove_stopwords(l_tokens)
		lsst_tokens = TokenUtils.remove_shortwords(lst_tokens, self.min_length)
		lwsst_tokens = TokenUtils.wordnet_lemmatizer(lsst_tokens)
		freqs = TokenUtils.get_token_frequencies(lwsst_tokens)
		for w in lwsst_tokens:	
			if freqs[w] >= self.min_occ and freqs[w] <= self.max_freq*len(lwsst_tokens):	
				doc_tokens.append(w)
		return doc_tokens

	def get_token_stats(self):
		t_stats = {}
		freqs = TokenUtils.get_token_frequencies(self.tokens)
		for w in self.tokens:	
			indices = TokenUtils.get_token_indices(w, self.tokens)
			t_stats[w] = TokenStats(w,freqs[w],indices)
		return freqs, t_stats

	def get_sorted_token_set(self):
		st = sortedset()
		return st.union(self.tokens)

	def get_collocation_mat(self,normalize=False):
		""" compute the association matrix elements """
		l = len(self.token_set)
		if l == 0:
			return None
		# Row-based linked list sparse matrices
		vmat = sps.lil_matrix((l,l),dtype=float)
		cmat = sps.lil_matrix((l,l),dtype=int)
		for i in xrange(l):
			for j in xrange(l):
				if i != j:
					indices1 = self.token_stats[self.token_set[i]].indices
					indices2 = self.token_stats[self.token_set[j]].indices
					# get collocation distance and number of pairs
					vmat[i,j],cmat[i,j] = self.metrics.get_collocation_distance(indices1, indices2)
		# normalize vmat entries
		if normalize:
			total = cmat.sum()
			for i in xrange(l):
				for j in xrange(l):
					vmat[i,j] = self.metrics.g(vmat[i,j],cmat[i,j],total,*self.metrics.args2)	
		return AssociationMatrix(MatrixUtils.row_normalize(vmat),cmat,self.token_set)

	def print_highest_entries(self,mat,m):
		entries = {}
		#for i in range(len(mat)):
		for i in xrange(mat.shape[0]):
			for j in range(mat.shape[1]):
				if i != j:
					v = mat[i,j]
					if not entries.has_key(v):
						entries[v] = [] 
					entries[v].append(self.token_set[i] + ':' + self.token_set[j])	
		ske = entries.keys()
		ske.sort(reverse=True)	
		i = 0
		for e in ske:
			print e,entries[e]
			i = i + 1
			if i == m:
				break

if __name__ == '__main__':
	rgx = '\w+'
	#punct = '\',.!?'
	min_length = 3
	# min number of occurences
	min_occ = 5
	# max frequency (between 0 and 1)
	max_freq = 1
	# normalize entries
	normalize = True	
	
	#### test tokenizers
	#sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good. Arthur, morning"""

	#print sentence
	##print word_tokenizer(sentence)
	##print regex_tokenizer(sentence, rgx)
	##print wordpunct_tokenizer(sentence)

	#### test stop words removal
	#tokens = TokenUtils.regex_tokenizer(sentence, rgx)
	#print tokens
	#st_tokens = TokenUtils.remove_stopwords(tokens)
	#print st_tokens
	#sst_tokens = TokenUtils.remove_shortwords(st_tokens, 3)
	#print sst_tokens

	##tokens = wordpunct_tokenizer(sentence)
	##print remove_punct_tokens(tokens, punct)

	#### tets lemmatization
	#wsst_tokens = TokenUtils.wordnet_lemmatizer(sst_tokens)
	#print wsst_tokens
	#lwsst_tokens = TokenUtils.tokens_to_lowercase(wsst_tokens)
	#print lwsst_tokens

	#### test word stats
	#freq = TokenUtils.get_token_frequencies(lwsst_tokens)
	#print freq
	#freqd = TokenUtils.get_most_freq_tokens(freq, 2)
	#print freqd

	####
	#print TokenUtils.get_token_indices('arthur', lwsst_tokens)
	
	###
	txt = open(sys.argv[1], 'r').read()
	cmetrics = CollocationMetrics(CollocationMetrics.decreasing_exp,(1,1),CollocationMetrics.info,())
	doc = DocStats(txt, rgx, min_length, min_occ, max_freq, cmetrics)
	#print doc.get_sorted_token_set()
	#print doc.token_stats
	mat = doc.get_collocation_mat(normalize)
	doc.print_highest_entries(mat,20)
	
	#### compute PageRank
	density = 0.0025
	g = GraphUtils.get_graph_from_association_matrix(mat,doc.token_set,density)
	pr = g.pagerank()
	
	argmax = lambda lst: lst.index(max(lst))
	for i in range(10):
		index = argmax(pr)
		print doc.token_set[index]
		pr[index] = 0
		
