def follow(thefile):
    '''generator function that yields new lines in a file

    >>> logfile = open("run/foo/access-log","r")
    >>> loglines = follow(logfile)
    >>> # iterate over the generator
    >>> for line in loglines:
    >>>     print(line)
    '''
    import os
    import time

    # seek the end of the file
    thefile.seek(0, os.SEEK_END)
    
    # start infinite loop
    while True:
        # read last line of file
        line = thefile.readline()
        # sleep if file hasn't been updated
        if not line:
            time.sleep(1.0)
            continue

        yield line
def watch_log(logfile):
    import logging
    import time

    while True:
        try:
            logfile = open(logfile, "r")
            break
        except IOError as e:
            time.sleep(1.0)
    logging.info(f"======================= {logfile.name} Start =========================")
    # iterate over the generator
    for line in follow(logfile):
        print(line, end='')