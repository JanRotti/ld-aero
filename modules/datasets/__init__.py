from .mnist import MNISTDataModule
try:
    from .rae import RAE, RAEDataModule
except Exception as e:
    print(e)

try:
    from .hein_do import HeinDo, HeinDoDataModule
except Exception as e:
    print(e)