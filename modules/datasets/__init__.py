from .mnist import MNISTDataModule
try:
    from .rae import RAE, RAEDataModule
except e:
    print(e)

try:
    from .hein_do import HeinDo, HeinDoDataModule
except e:
    print(e)