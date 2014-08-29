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

from utils import *
import re
import sys
import cPickle as pickle

if __name__ == '__main__':
	""" copy <sample_size> male and female blogs """
	source_dir = sys.argv[1]	
	target_dir = sys.argv[2]	
	l = FileUtils.get_files_list(source_dir, '.*\(1\).*')
	rx = re.compile('.*\.male\..*')
	males = []
	females = []
	nmales = 0
	nfemales = 0
	ntotal = 0
	sample_size = 250

	print '# files: ', len(l)

	for p in l:
		if rx.match(p):
			males.append(p)
		else:
			females.append(p)

	while ntotal < sample_size*2 and (len(males)>0 or len(females)>0):
		if len(males)>0 and (nmales < sample_size or (len(females) == 0 and ntotal < sample_size*2)):
			p = males.pop()
			p2 = p.replace('(1)', '(2)')
			md = p.replace('(1)', '(main-doc)')
			dp1 = p.replace('(1)', '(doc-part1)')
			mp1 = p.replace('(1)', '(mat-part1)')
			FileUtils.copy_file(source_dir,target_dir,p)
			FileUtils.copy_file(source_dir,target_dir,p2)
			FileUtils.copy_file(source_dir,target_dir,md)
			FileUtils.copy_file(source_dir,target_dir,dp1)
			FileUtils.copy_file(source_dir,target_dir,mp1)
			nmales += 1
			ntotal += 1
		if len(females) and (nfemales < sample_size or (len(males) == 0 and ntotal < sample_size*2)):
			p = females.pop()
			p2 = p.replace('(1)', '(2)')
			md = p.replace('(1)', '(main-doc)')
			dp1 = p.replace('(1)', '(doc-part1)')
			mp1 = p.replace('(1)', '(mat-part1)')
			FileUtils.copy_file(source_dir,target_dir,p)
			FileUtils.copy_file(source_dir,target_dir,p2)
			FileUtils.copy_file(source_dir,target_dir,md)
			FileUtils.copy_file(source_dir,target_dir,dp1)
			FileUtils.copy_file(source_dir,target_dir,mp1)
			nfemales += 1
			ntotal += 1
		print '# males: ', nmales, '# females: ', nfemales
