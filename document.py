class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        self.result=float('inf')
        if amount==0:
            return 0
        dic={}
        def coin(amount):
            if amount==0:
                return 0
            if amount in dic:
                return dic[amount]
            k=float('inf')
            for i in coins:
                if i<=amount:
                    k=min(k,1+coin(amount-i))# instead of p we are doing something like 1+ to know the path
            dic[amount]=k
            return dic[amount]
        self.result=coin(amount)
        if self.result==float('inf'):
            return -1
        return self.result
#when we are trying something like this first we should do like line 7 and line 8 thinking about the ground case and initialzing

# for memoization use something like k=float("inf") when we are doing something like for i in coins(recursion)
