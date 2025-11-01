import os                                                                       
from multiprocessing import Pool                                                
                                                                                

# 4 workers
processes = (
    'pgelib/redis_ako.py wk 0',
    'pgelib/redis_ako.py wk 1',
    'pgelib/redis_ako.py wk 2',
    'pgelib/redis_ako.py wk 3'
)


def run_process(process):

    os.system('python {}'.format(process))                                       


# 4 workers
pool = Pool(processes=4)
pool.map(run_process, processes)


