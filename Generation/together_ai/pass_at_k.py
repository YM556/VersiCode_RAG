import math

def pass_at_k(n, c, k):
    if n-c<k:
        return 1.0
    print("n:{},c:{},k:{}".format(n, c, k))
    score = 1 - (math.comb(n - c, k)) / (math.comb(n, k))

    return score

for k in [1,3,10]:
    print(pass_at_k(10,1,k))