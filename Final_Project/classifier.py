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

	print(article_list)
	for article in article_list:
		print(article[3])


if __name__ == "__main__":
	main()
