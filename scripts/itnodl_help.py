#  ___        _   _
# |_ _|_ _   | |_| |_  ___   _ _  __ _ _ __  ___
#  | || ' \  |  _| ' \/ -_) | ' \/ _` | '  \/ -_)
# |___|_||_|  \__|_||_\___| |_||_\__,_|_|_|_\___|    _
#  ___ / _|  __| |___ ___ _ __  | |___ __ _ _ _ _ _ (_)_ _  __ _
# / _ \  _| / _` / -_) -_) '_ \ | / -_) _` | '_| ' \| | ' \/ _` |
# \___/_|   \__,_\___\___| .__/ |_\___\__,_|_| |_||_|_|_||_\__, |
#                        |_|                               |___/

"""
HELPER FUNCTIONS

Coded by Wouter Durnez
-- Wouter.Durnez@UGent.be
-- Wouter.Durnez@student.kuleuven.be
"""

import os
import time

# GLOBAL VARIABLES #
####################

# Set log level (1 = only top level log messages -> 3 = all log messages)
LOG_LEVEL = 1


# FUNCTIONS #
#############

def log(*message, lvl=3, sep="", title=False):
    """Print wrapper, adds timestamp."""

    # Set timezone
    if 'TZ' not in os.environ:
        os.environ['TZ'] = 'Europe/Amsterdam'
        time.tzset()

    # Title always get shown
    lvl = 1 if title else lvl

    # Print if log level is sufficient
    if lvl <= LOG_LEVEL:

        # Print title
        if title:
            n = len(*message)
            print('\n' + (n + 4) * '#')
            print('# ', *message, ' #', sep='')
            print((n + 4) * '#' + '\n')

        # Print regular
        else:
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(str(t), (" - " if sep == "" else "-"), *message, sep=sep)

    return


def time_it(f):
    """Timer decorator: shows how long execution of function took."""

    def timed(*args, **kwargs):
        t1 = time.time()
        res = f(*args, **kwargs)
        t2 = time.time()

        log("\'", f.__name__, "\' took ", round(t2 - t1, 3), " seconds to complete.", sep="")

        return res

    return timed


def make_folders(*folders):
    """
    If folders don't exist, make them.
    :param folders:
    :return: None
    """

    for folder in folders:
        if not os.path.exists(os.path.join(os.pardir, folder)):
            os.makedirs(os.path.join(os.pardir, folder))
            log("Created \'", folder, "\' folder.", lvl=3)
        else:
            log("\'{}\' folder accounted for.".format(folder), lvl=3)


# MAIN #
########

if __name__ is "__main__":
    log("Behold, peasants", title=True)

    log("Nothing to see here, move along.", lvl=1)
    log("No really, nothing.", lvl=2)
    log("...or is there?.", lvl=3)