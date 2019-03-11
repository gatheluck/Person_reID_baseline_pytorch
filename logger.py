import os
import pandas
from datetime import datetime


__all__ = [
	'Logger',
	'get_time_stamp'
]


class Logger():
	def __init__(self, path, num_epochs, time_stamp=None):
		self.path = path
		self.num_epochs = num_epochs
		if time_stamp is None:
			self.time_stamp = get_time_stamp()
		else:
			self.time_stamp = time_stamp

		if os.path.exists(path):
			self.df = pandas.read_csv(path, index_col=0)
		else:
			self.df = pandas.DataFrame(index=range(1,num_epochs+1)) # range is from 1 to (num+1)

		self.df[self.time_stamp] = [None] * num_epochs
		self.save()

	def set(self, epoch, val):
		self.df[self.time_stamp][epoch] = val
		print(val)
		self.save()

	def save(self):
		self.df.to_csv(self.path)


def get_time_stamp():
	return datetime.now().strftime('%Y/%m/%d %H:%M:%S')


if __name__ == "__main__":
	log = Logger('./logs/logger_test.csv', 10)
	for i in range(1,10):
		log.set(i, i)
		print(log.df)