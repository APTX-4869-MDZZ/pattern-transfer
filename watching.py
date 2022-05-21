import psutil
import time
import os

def watching(current_task, next_task):
  while True:
    running = False
    for pid in psutil.pids():
      p = psutil.Process(pid)
      if p.name() == 'python.exe':
        if current_task in p.cmdline():
          running = True
          print('{} {} still running...'.format(time.asctime(time.localtime(time.time())), current_task))
          break
    if not running:
      print('start next task: ' + next_task)
      os.system('python ' + next_task)
      break
    time.sleep(60*20)

watching('downloadImage.py', 'downloadImage.py')