from dataclasses import dataclass


@dataclass
class X:
    x: int = 1


@dataclass
class Y(X):
    x: int = 2


y = Y().x
print(y)
