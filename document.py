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



class Trie:

    def __init__(self):
        self.dic={}


    def insert(self, word: str) -> None:
        cur=self.dic

        for letter in word: #it's all about playing with dictionary
            if letter not in cur:
                cur[letter]={} # here we are trying to add one dic
            cur=cur[letter] #here one step further
        cur['*']='' # end of text
        

    def search(self, word: str) -> bool:
        cur=self.dic
        for i in word:
            if i not in cur:
                return False
            cur=cur[i]
        if '*' not in cur: return False
        return True
        

    def startsWith(self, prefix: str) -> bool:
        cur=self.dic
        for i in prefix:
            if i not in cur:
                return False
            cur=cur[i]
        return True
        


# Your Trie object will be instantiated and called as such:



# instead of doing like stack add in between or , and
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

found=dfs(word+board[i][j],visited,i-1,j,actualWord) or dfs(word+board[i][j],visited,i+1,j,actualWord) or dfs(word+board[i][j],visited,i,j-1,actualWord) or dfs(word+board[i][j],visited,i,j+1,actualWord)
Not like
dfs(word+board[i][j],visited,i-1,j,actualWord)
dfs(word+board[i][j],visited,i+1,j,actualWord)
dfs(word+board[i][j],visited,i,j-1,actualWord)
dfs(word+board[i][j],visited,i,j+1,actualWord)


