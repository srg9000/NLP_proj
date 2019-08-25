import pandas as pd
import numpy as np
import re


max_len = 0
s = set()
fp = open('itemset_word_demo.csv', 'w+')
with open('itemset_small_full.csv', 'r') as f:
	while(1):
		try:
			l = f.readline()
#			l = re.sub("\'","",l)
			
			
			if len(l) == 0:
				break
			l = l.split(',')
			[l.remove(i) for i in l if len(i)==1 and i!='i']
			if len(l)>5:# and int(l[-1])>80:
				print(*l, file = fp, sep = ',', end = '\n')
#				print(*l, sep = ',', end = '\n')
#			s.add(int(l[-1]))
	#			print(s)
			if len(l)>max_len:
				max_len = len(l)
#				print(l)
			
		except:
			fp.close()
			f.close()
			break

#print(max(s), max_len)
fp.close()
f.close()