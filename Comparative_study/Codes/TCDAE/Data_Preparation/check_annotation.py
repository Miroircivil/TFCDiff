import glob
import numpy as np
from scipy.signal import resample_poly
import wfdb
import math
import _pickle as pickle


def prepare(QTpath='data/qt-database-1.0.0/'):
    # Desired sampling frequency
    newFs = 360

    # Preprocessing signals
    namesPath = glob.glob(QTpath + "/*.dat")

    # final list that will contain all signals and beats processed
    QTDatabaseSignals = dict()

    # Open the file in write mode
    with open('log.txt', 'w', encoding='utf-8') as log_file:
        register_name = None
        for i in namesPath:
            # reading signals
            aux = i.split('.dat')
            register_name = aux[0].split('/')[-1]
            signal, fields = wfdb.rdsamp(aux[0])
            qu = len(signal)

            # reading annotations
            ann = wfdb.rdann(aux[0], 'pu1')

            # Only focus on the 'symbol', 'sample', and 'num' attributes
            log_file.write(f"For the 'pu1' annotation of record {register_name}:\n")
            attributes = ['symbol', 'sample', 'num']
            for attr in attributes:
                if hasattr(ann, attr):
                    value = getattr(ann, attr)
                    log_file.write(f"  {attr}: {type(value)}\n")
                    if isinstance(value, (list, np.ndarray)):
                        log_file.write(f"    Length: {len(value)}\n")
                        if attr == 'symbol':
                            unique_symbols = set(value)
                            log_file.write(f"    Number of different annotation types: {len(unique_symbols)}\n")
                            log_file.write(f"    Specific annotations: {', '.join(sorted(unique_symbols))}\n")
                    elif isinstance(value, dict):
                        log_file.write(f"    Number of keys: {len(value.keys())}\n")
                else:
                    log_file.write(f"  The 'ann' object does not have the {attr} attribute.\n")


if __name__ == "__main__":
    prepare()
