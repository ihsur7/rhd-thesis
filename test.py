import math
# t = 1
# while t > 0:
#     conc_ratio = math.erfc(1/math.sqrt(4*67.6e-5*t))
#     if conc_ratio == 1:
#         print(t)
#         break
#     else:
#         t += 10000000
t = 1e8
conc_ratio = math.erfc((14*(0.25*35)/math.sqrt(4*67.6e-5*t)))
print(conc_ratio)