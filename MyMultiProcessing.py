from logging import error
from multiprocessing import Pool, Manager
from itertools import repeat
import multiprocessing
from tqdm import tqdm
import threading
import time

class MyMultiProcessing:
  def __init__(self, pool_num, function, iterations, **args):
    m = Manager()
    self.error_file = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime()) + ' error.log'
    self.error_queue = m.Queue()
    self.pool_num = pool_num
    self.total = args.get('total', len(iterations))
    self.ordered = args.get('ordered', False)
    self.returned = args.get('returned', False)
    if self.returned:
      self.parse_return_result(args, m)
      iterations = zip(repeat(self.result), iterations)
    self.iterations = zip(repeat(function), repeat(self.error_queue), range(self.total), iterations)

    self._error_handler = threading.Thread(
      target=MyMultiProcessing.write_error_to_file,
      args=(self.error_queue, self.error_file)
    )
    self._error_handler.daemon = True
    self._error_handler.start()
    
  def parse_return_result(self, args, m):
    self.result_type = args.get('result_type', 'value')

    if self.result_type == 'value':
      self.operate = args.get('operate', 'sum')
      self.result = m.Value('result', args.get('init_result', 0))
    elif self.result_type == 'dict':
      self.result = m.dict(args.get('init_result', {}))
    elif self.result_type == 'list':
      self.result = m.list(args.get('init_result', []))
    if 'callback_func' in args:
      self.callback_func = args['callback_func']
    if 'callback_args' in args:
      self.callback_args = args['callback_args']

  def start(self):
    with Pool(self.pool_num) as pool:
      if self.ordered:
        list(tqdm(pool.imap(MyMultiProcessing.catchError, self.iterations), total=self.total))
      else:
        list(tqdm(pool.imap_unordered(MyMultiProcessing.catchError, self.iterations), total=self.total))
    pool.close()
    pool.join()
    if hasattr(self, 'callback_func'):
      if hasattr(self, 'callback_args'):
        self.callback_args.insert(0, self.result)
      else:
        self.callback_args = [self.result]
      self.callback_func(*self.callback_args)

  @staticmethod
  def catchError(args):
    func = args[0]
    error_queue = args[1]
    try:
      if isinstance(args[3], tuple):
        return func(*args[3])
      else:
        return func(args[3])
    except Exception as e:
      error_queue.put('index ' + str(args[2]) + ': ' + str(e) + '\n')

  @staticmethod
  def write_error_to_file(error_queue, error_file):
    with open(error_file, 'w', encoding='utf-8') as f:
      while 1:
        try:
          e = error_queue.get()
          f.write(e)
        except:
          return

def print_i(i):
  with open('test/{}.txt'.format(i), 'w') as f:
    f.write(str(i))

def print_position(i, x, y):
  with open('test/{}.txt'.format(i), 'w') as f:
    f.write(str(x) + ' ' + str(y))

def create_dict(result_dict, i):
  result_dict[i] = i**2

def print_dict(result_dict, key_list):
  for key in key_list:
    print(str(key) + ' ' + str(result_dict[key]))

if __name__=='__main__':
  ### with callback process function ###
  multiprocessing = MyMultiProcessing(
      4, create_dict, range(20), # 进程数, 进程执行函数, 函数参数列表
      total=20, ordered=False, # total: 运行总数（可视化进度条用）, 是否需要有序
      returned=True, result_type='dict', init_result={}, # 是否有返回值, 返回值类型[value, dict, list], 初始化结果
      callback_func=print_dict, callback_args=[range(20)] # 返回值后处理函数
  )
  multiprocessing.start()

  ### w/o callback process function ###
  # multiprocessing = MyMultiProcessing(
  #   4, print_i, range(10),
  #   total=10, ordered=False
  # )
  # multiprocessing.start()
