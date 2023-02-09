from multiprocessing import Process, Lock

def func(l, i, range):
    l.acquire()
    try:
        print(f'Hello, I am the {i}th of {range} processes')
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()

    for num in range(10):
        Process(target=func, args=(lock, num, 4)).start()
