from typing import List
from datetime import datetime

class Logger:
    def __init__(self, file: str, 
                       include: List[str], 
                       stdout: bool):
        self.file_str = file
        self.file = None
        self.include = include
        self.stdout = stdout

    def log(self, type, msg):
        out = f'{datetime.now()} | {type} | {msg}'
        if type in self.include:
            if self.stdout:
                print(out)
        self.file.write(out + '\n')

    def open(self):
        self.file = open(self.file_str, 'w')
    def close(self):
        self.file.close()