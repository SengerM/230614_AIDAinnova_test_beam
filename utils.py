import pandas
from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import numpy
import subprocess
import datetime
from contextlib import nullcontext
from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
import logging

PATH_TO_WHEREVER_THE_DOCKER_DATA_IS_POINTING_TO = Path('/home/msenger/240217_DESY_test_beam')

PLOTS_LABELS = {
	'DUT_name_rowcol': 'DUT (i,j)',
	'Px': 'x (m)',
	'Py': 'y (m)',
	'cluster_size': 'Cluster size',
	'efficiency': 'Efficiency',
	'efficiency (%)': 'Efficiency (%)',
	'efficiency_error': 'Efficiency error',
	'efficiency_error (%)': 'Efficiency error (%)',
}

def save_dataframe(df, name:str, location:Path):
	for extension,method in {'pickle':df.to_pickle,'csv':df.to_csv}.items():
		method(location/f'{name}.{extension}')

def select_by_multiindex(df:pandas.DataFrame, idx:pandas.MultiIndex)->pandas.DataFrame:
	"""Given a DataFrame and a MultiIndex object, selects the entries
	from the data frame matching the multi index. Example:
	DataFrame:
	```
	       d
	a b c   
	1 9 4  1
	2 8 2  4
	3 7 5  2
	4 6 4  5
	5 5 6  3
	6 4 5  6
	7 3 7  4
	8 2 6  7
	9 1 8  5
	```
	MultiIndex:
	```
	MultiIndex([(1, 9),
            (2, 8),
            (3, 7),
            (4, 6),
            (9, 1)],
           names=['a', 'b'])
	```
	Output:
	```
	       d
	a b c   
	1 9 4  1
	2 8 2  4
	3 7 5  2
	4 6 4  5
	9 1 8  5

	```
	"""
	if not set(idx.names) <= set(df.index.names):
		raise ValueError('Names in `idx` not present in `df.index`')
	if not isinstance(df, pandas.DataFrame) or not isinstance(idx, pandas.MultiIndex):
		raise TypeError('`df` or `idx` are of the wrong type.')
	lvl = df.index.names.difference(idx.names)
	return df[df.index.droplevel(lvl).isin(idx)]

def get_datanode_directory_within_corry_docker(datanode:DatanodeHandler):
	"""Get the absolute path of the datanode directory within the corry docker
	container."""
	TB_dn = datanode
	while True:
		if TB_dn.parent is None:
			break
		else:
			TB_dn = TB_dn.parent
	p = Path('/data')/datanode.path_to_datanode_directory.relative_to(TB_dn.path_to_datanode_directory.parent)
	return p

def run_commands_in_docker_container(command, container_id:str, stdout=None, stderr=None):
	"""Runs one or more commands inside a docker container.
	
	Arguments
	---------
	command: str or list of str
		A string with the command, or a list of strings with multiple
		commands to be executed sequentially.
	"""
	timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
	temp_file = PATH_TO_WHEREVER_THE_DOCKER_DATA_IS_POINTING_TO/f'.{timestamp}_{numpy.random.rand()*1e9:.0f}.sh'
	if not isinstance(command, (str, list)):
		raise TypeError(f'`command` must be a list or a string, received object of type {type(command)}')
	if isinstance(command, str):
		command = [command]
	if any([not isinstance(_, str) for _ in command]):
		raise TypeError(f'`command` must be a list of strings, but at least one element is not.')
	try:
		with open(temp_file, 'w') as ofile:
			print('#!/bin/bash', file=ofile)
			for c in command:
				print(c, file=ofile)
		subprocess.run(['chmod','+x',str(temp_file)])
		result = subprocess.run(
			['docker','exec','-it',container_id,f'/data/{temp_file.name}'],
			stdout = stdout,
			stderr = stderr,
		)
	except:
		raise
	finally:
		temp_file.unlink()
	return result

# ~ class DatanodeHandlerSubclassMe(DatanodeHandler):
	# ~ my_class = 'write here the class of your datanodes'
	# ~ parent_class = 'class of the parent'
	# ~ subdatanode_classes = {'task_1': 'class of subdatanodes of task 1', 'task_2': 'class of subdatanodes of task 2'}
	
	# ~ def __init__(self, path_to_datanode:Path):
		# ~ super().__init__(path_to_datanode, check_datanode_class=self.my_class)
	
	# ~ @property
	# ~ def parent(self):
		# ~ return super().parent.as_type(self.parent_class)
	
	# ~ def list_subdatanodes_of_task(self, task_name:str):
		# ~ subdatanodes = super().list_subdatanodes_of_task(task_name)
		# ~ if task_name in self.subdatanode_classes:
			# ~ subdatanodes = [_.as_type(self.subdatanode_classes[task_name]) for _ in subdatanodes]
		# ~ return subdatanodes
