from .mnist import MNISTDataModule
try:
    from .rae import RAE, RAEDataModule
except e:
    print(e)