# !/usr/bin/env python
# -*- coding:utf-8 -*- 

age_init = 22
money_init = 16.0
ratio_1 = 1.05  # 每年增加10%
ratio_2 = 1.05
ratio_t = 0.96

# 毕业——自主
money = [money_init]
for age in range(age_init+1, age_init + 21):
    money.append(money[-1] * ratio_1 * ratio_t)
    # print(money[-1])
    
# 自主——挂掉
money_init_2 = money[-1] * (1 - 2.4 / money_init) * 0.8
money.append(money_init_2)
for age in range(age_init + 20 + 1, 93):
    money.append(money[-1] * ratio_2 * ratio_t)
    # print(money[-1])
print('自主前：')
print(money[0:21])
print('自主后：')
print(money[21:71])

total_money = sum(money)
print('总数 = ',total_money)

sum = 100
for i in range(0, 21):
    sum = sum / 0.96
    
print('sum = ', sum)