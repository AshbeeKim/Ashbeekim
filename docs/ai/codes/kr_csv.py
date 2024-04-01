import os
import glob
# import re
# dirname = re.search('[^/]+$', os.getcwd()).group(0)
import numpy as np
import pandas as pd

class WIN_CSV:
    def __init__(self, day_num, check_num):
        self.dir_path = os.getcwd()
        self.data_dir = os.path.join(self.dir_path, f"data/day_{day_num}")
        self.csv_list = glob.glob(self.data_dir + "/*.csv")
        self.check_num = check_num
        self.data, self.col_list = self.preprocess()
    
    def preprocess(self):
        self.data = pd.read_csv(self.csv_list[self.check_num], encoding="euc-kr")
        self.col_list = self.data.columns
        return self.data, self.col_list
    
    def return_blen(self, col_type, srs):
        if col_type == object:
            self.blens = [len(blen) for blen in srs]
        elif col_type == float:
            blens = [str(blen) for blen in srs]
            self.blens = [len(blen) if blen[:-2]!=".0" else len(blen[:-2]) for blen in blens]
        else:
            srs = srs.apply(lambda x: str(x))
            srs = [num if ((num != "1.0") or (num != "0.0")) else num[:-2] for num in srs]
            self.blens = [len(blen.encode()) for blen in srs]
        return self.blens

    def fnum_bnum(self):

        self.fnum_col = []
        self.bnum_col = []
        return self.fnum_col, self.bnum_col

    def print_kr_csv(self):
        data = self.data
        col_list = self.col_list
        
        col_dtype = {cols:[] for idx, cols in enumerate(col_list)}
        col_blen ={cols:[] for idx, cols in enumerate(col_list)} 
        
        fnum_col = {}
        bnum_col = {}

        for cols in col_list:
            srs = data.loc[:,cols]
            col_dtype[cols] = "object" if srs.dtype=='O'else "numeric"

            blens = self.return_blen(srs.dtype, srs)
            col_blen[cols].append(np.min(np.array(blens)))
            col_blen[cols].append(np.max(np.array(blens)))

            if srs.dtypes == int:
                fnum = [len(str(num).split(',')[0]) for num in srs]
                bnum = [len(str(num).split(',')[1]) for num in srs]

                fnum_col[cols] = list(np.unique(np.array(fnum)))
                bnum_col[cols] = list(np.unique(np.array(bnum)))
            
            elif srs.dtypes == float:
                fnum = [len(str(num).split('.')[0]) for num in srs]
                bnum = [len(str(num).split('.')[1]) for num in srs]

                fnum_col[cols] = list(np.unique(np.array(fnum)))
                bnum_col[cols] = list(np.unique(np.array(bnum)))

        print(f'''\tDATA TYPE : \n{pd.Series(col_dtype).to_markdown()}\
            \n\tBYTE LENGTH : \n{pd.Series(col_blen).to_markdown()}\
            \n\tFNUM LENGTH(MIN,MAX) : \n{pd.Series(fnum_col).to_markdown()}\
            \n\tBNUM LENGTH(MIN,MAX) : \n{pd.Series(bnum_col).to_markdown()}''')
