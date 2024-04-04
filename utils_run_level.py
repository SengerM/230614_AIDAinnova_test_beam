from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import numpy
import pandas
import sqlite3
import logging
from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
from contextlib import nullcontext

def execute_EUDAQ_run_task_on_all_runs_within_batch(TB_batch_dn:DatanodeHandler, func:callable, args:dict, telegram_bot_reporter:SafeTelegramReporter4Loops=None):
	TB_batch_dn.check_datanode_class('TB_batch')
	
	EUDAQ_runs_to_be_processed = TB_batch_dn.list_subdatanodes_of_task('EUDAQ_runs')
	
	with telegram_bot_reporter.report_loop(
		total_loop_iterations = len(EUDAQ_runs_to_be_processed),
		loop_name = f'{func.__name__} on {TB_batch_dn.pseudopath}',
	) if telegram_bot_reporter is not None else nullcontext() as reporter:
		for EUDAQ_run_dn in EUDAQ_runs_to_be_processed:
			logging.info(f'Running `{func.__name__}` on {EUDAQ_run_dn.pseudopath}...')
			func(EUDAQ_run_dn, **(args[EUDAQ_run_dn.datanode_name]))
			reporter.update(1) if telegram_bot_reporter is not None else None
	logging.info(f'Finished running `{func.__name__}` on all EUDAQ_runs within TB_batch `{TB_batch_dn.pseudopath}` ✅')
