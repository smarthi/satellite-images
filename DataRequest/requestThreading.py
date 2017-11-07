'''
Script which enables threading of function over list of inputs
'''

import threading
try:
    import queue
except ImportError:
    import Queue as queue

DEFAULT_THREAD_NUMBER = 8

class ThreadedProcess():
    def __init__(self, input_list, process_function, thread_number=DEFAULT_THREAD_NUMBER):
        self.input_list = input_list
        self.process_function = process_function
        self.thread_number = thread_number

        self.output_list = [None] * len(self.input_list)
        self.threading_done = False

    def run_threading(self):
        my_queue = queue.Queue()
        threads = []
        for i in range(self.thread_number): # initializing threads
            threads.append(MyThread(my_queue=my_queue, output_list=self.output_list, process_function=self.process_function))
            threads[-1].start()

        for i in range(len(self.input_list)):
            my_queue.put((self.input_list[i], i))

        my_queue.join() # waits until all threads are done
        for i in range(self.thread_number):
            my_queue.put(None)
        for thread in threads:
            thread.join()
        self.threading_done = True

    def get_output(self):
        if not self.threading_done:
            self.run_threading()
        return self.output_list

class MyThread(threading.Thread):
    def __init__(self, my_queue, output_list, process_function):
        threading.Thread.__init__(self)
        self.my_queue = my_queue
        self.output_list = output_list
        self.process_function = process_function

    def run(self):
        while True:
            input_request = self.my_queue.get()
            if input_request is None:
                break
            input_item, input_index = input_request
            self.output_list[input_index] = self.process_function(input_item)
            self.my_queue.task_done()
