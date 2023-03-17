import numpy as np

ar=np.array([[ 85.3536, -53.0540,  47.1546],
    [-53.0636,  74.8130,  -9.4000],
    [ 35.4160, -10.5083,  44.7809]])
u,s,vh = np.linalg.svd(ar)
print(f'u={u}')
print(f's={s}')
print(f'vh={vh}')
