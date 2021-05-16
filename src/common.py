# This file houses all the functions that are needed by multiple files


# Common function header

# def funct(a: type, b: type, c: type) -> type:
#     '''
#       Short description of the function

#       Args:
#         a: short desc
#         b: short desc
#         c: short desc

#       Returns:
#         return value : short desc
#     '''

from datetime import date, datetime

def load_meta(file):
    '''
    Load the information from the file 'meta.txt' and return a dictionry with the corresponding metadata and its values

    Args: 
        None
    Returns:
        Dict[str,str]: a dictionary of the meta parameters
    '''
    f = open(file, 'r')
    out = {}
    for line in f.readlines():
        s = line.strip().split(' ')
        out[s[0]] = s[1]
    f.close()
    return out

def log(type, msg):
    print(f'{datetime.now()} | {type} | {msg}')