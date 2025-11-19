import numpy as np
with open('c.txt','w') as f:
    for r in np.arange(1.326,4.5,0.2):
        f.write(f'qsub runit.sh {r} \n')
