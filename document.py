class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        self.result=float('inf')
        if amount==0:
            return 0
        dic={}
        for i in coins:
            dic[i]=1
        def coin(amount):
            if amount==0:
                return 0
            if amount in dic:
                return dic[amount]
            k=float('inf')
            for i in coins:
                if i<=amount:
                    k=min(k,1+coin(amount-i))
            dic[amount]=k
            return dic[amount]
        self.result=coin(amount)
        if self.result==float('inf'):
            return -1
        return self.result

# for memoization use something like k=float("inf") when we are doing something like for i in coins(recursion)
