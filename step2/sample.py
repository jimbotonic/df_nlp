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

from utils import *
import re
import sys
import cPickle as pickle

if __name__ == '__main__':
	""" copy <sample_size> male and female blogs """
	source_dir = sys.argv[1]	
	target_dir = sys.argv[2]	
	l = FileUtils.get_files_list(source_dir)
	rx = re.compile('.*\.male\..*')
	males = []
	females = []
	sample_size = 500
	male_OK = False
	female_OK = False

	for p in l:
		if rx.match(p):
			if len(males) < sample_size:
				FileUtils.copy_file(source_dir,target_dir,p)
				males.append(p)
			else:
				male_OK = True
		else:
			if len(females) < sample_size:
				FileUtils.copy_file(source_dir,target_dir,p)
				females.append(p)
			else:
				female_OK = True
		if male_OK and female_OK:
			break
