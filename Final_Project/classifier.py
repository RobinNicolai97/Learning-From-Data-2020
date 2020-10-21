import json
import os


def read_corpus(dir_name='data'):
	file_list = os.listdir(dir_name)
	article_list = []

	for file in file_list:
		with open(dir_name + "\\" + file) as f:
			data = json.load(f)
			articles = data['articles']
			for article in articles:
				article_list.append(article)

	return article_list


def main():
	article_list = read_corpus('data')
	subjects = []
	#print(article_list)
	print('number of articles:', len(article_list))
	for article in article_list:
		#print(article[3])
		subs = article['classification']['subject']
		if subs:
			for sub in subs:
				subjects.append(sub['name']) 
		else: 
			subjects.append('ZNONE') 
	#subjects = [sub for sub in subjects if sub.isupper()]
	set_subs = set(subjects)
	set_subs = sorted(set_subs)
	f = open("noisy_subjects.txt", "w")
	for sub in set_subs:
		#print(sub, subjects.count(sub)) 
		#if subjects.count(sub) == 1:
		f.write(str(sub)+ ' '+ str(subjects.count(sub))+'\n')
	f.close()
if __name__ == "__main__":
	main()
