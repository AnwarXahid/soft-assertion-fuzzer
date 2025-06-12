import torch
from softassertion.analysis.boundary_tracer import start_fuzz, end_fuzz

def test():
    start_fuzz()

    x = torch.rand((3, 3))
    y = torch.nn.functional.softmax(x, dim=1)
    print("Softmax output:\n", y)

    z = torch.rand((3, 3)) - 0.5  
    w = torch.log(z)
    print("Log output:\n", w)

    end_fuzz()

if __name__ == '__main__':
    test()
