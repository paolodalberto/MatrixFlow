from datetime import datetime
import os.path

started = datetime.now()


# ======================================================================
#
#  Functions in alphabetical order
#


def check(cond, msg):
    """Aborts, if condition is not met"""
    if not cond:
        o()
        o("Condition violated:")
        o(msg)
        fatal("program aborted")


def date_stamp():
    """  Return current date/time as string """
    return datetime.now().ctime()


def exists(file_name):
    """Returns true iff fileName belongs to an existing file"""
    return os.path.exists(file_name) and os.path.isfile(file_name)


def fatal(msg):
    """Closes program after fatal problem"""
    o()
    o("Program abort after fatal problem")
    o(msg)
    o()
    finish(1)


def find(value_list, value):
    """Search list for value; return index iff found, -1 otherwise"""
    if value in value_list:
        return value_list.index(value)
    else:
        return -1


def finish(return_code):
    """Program exit"""
    global started
    elapsed = round(float(str((datetime.now() - started).total_seconds())), 2)
    o()
    o(f"Elapsed: {elapsed}s")
    exit(return_code)


def o(s=""):
    """Central text output"""
    print(s)


def pretty_num(n):
    """Return number formatted with thousands separator if needed"""
    if n < 0:
        return f"-{pretty_num(-n)}"
    if n < 1000:
        return str(n)
    return f"{pretty_num(n // 1000)},{str(n % 1000).ljust(3, '0')}"
