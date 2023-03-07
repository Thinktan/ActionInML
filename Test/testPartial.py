from functools import partial

def test_func(a, b, c):
    return a, b, c

if __name__ == '__main__':
    fn = partial(test_func, b=1, c=2)
    res_fn = fn(a=4)
    print(res_fn)