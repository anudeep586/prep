#heap when you think heap think about insert only necessary k values in it and pop and do it don't insert all and do it. always think about heap and hashset or map

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
                if i<=amount: #always keep a condition after for loop it is good actually trust me
                    k=min(k,1+coin(amount-i))# instead of p we are doing something like 1+ to know the path if we are adding here return o from top(pattern)
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
OR
directions = [(1,0),(0,1),(-1,0),(0,-1)]
for x,y in directions:
    if 0<=i+x<=rows-1 and 0<=j+y<=cols-1:
        dfs(i+x,j+y,currWord+board[i+x][j+y],root.children[currLetter])




class TrieNode:
    
    #This is one node of the big Trie that we're gonna make 
    
    def __init__(self):
        self.children = {}  #This is the first level, all the starting letters basically
        self.end = False    #This will be set to true for a child if that words exists in the Trie
class Trie:
    def __init__(self):
        self.rootNode = TrieNode()      #Our root node for the main try

    def insert(self, word):
        
        start = self.rootNode           #Basic insertion of words in Tries
        for letter in word:
            if letter not in start.children:
                start.children[letter] = TrieNode()
            start = start.children[letter]
        start.end = True                #Notice this is actually start.children[letter] = True
                                        #start.children[letter] will contain all the words which has a prefix up until that letter
                                        #If start.children[letter].end is true it means the words exists
    def getRoot(self):
        return self.rootNode
    

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        
        def dfs(i, j, currWord, root):
            currLetter = board[i][j]
            
            if currLetter not in root.children:             #End the dfs call if the remaining word is absent
                return 
            
            board[i][j] = "*"                               #So that we do not visit the same letter in the matrix again
            if root.children[currLetter].end:
                output.add(currWord)                        #Should not return the dfs call, continue because [road,roadster]
            
            for x,y in directions:
                if 0<=i+x<=rows-1 and 0<=j+y<=cols-1:
                    dfs(i+x,j+y,currWord+board[i+x][j+y],root.children[currLetter])
            
            board[i][j] = currLetter                        #Restroring the letter in the Matrix
        
        rows = len(board)
        cols = len(board[0])
        
        myTrie = Trie()
        for word in words:                                  #Insertion of words into the tree
            myTrie.insert(word)
        
        rootNode = myTrie.getRoot()

        output = set()
        
        directions = [(1,0),(0,1),(-1,0),(0,-1)]
        for i in range(rows):
            for j in range(cols):
                dfs(i,j,board[i][j],rootNode)
            
        return output
        

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        num = {
            '1':'',
            '2':'abc',
            '3':'def',
            '4':'ghi',
            '5':'jkl',
            '6':'mno',
            '7':'pqrs',
            '8':'tuv',
            '9':'wxyz',
        }
        self.res=[]
        if len(digits)==0:
            return []
        def dfs(i,p):
            if len(p)==len(digits):
                self.res.append(p)
                return 
            for k in num[digits[i]]:
                dfs(i+1,p+k)
            return
        dfs(0,'')
        print(self.res)
        return self.res
        
# when we are storing the visited nodes then keep track of nodes like 

k=float('inf') # before
for i in coins:
    if i<=amount:
        k=min(k,1+coin(amount-i))# instead of p we are doing something like 1+ to know the path
dic[amount]=k # after

or 

def dfs(i,p):
    if len(p)==len(digits):
        self.res.append(p)
        return 
    for k in num[digits[i]]:
        dfs(i+1,p+k)
    return

or re-leave the nodes

visited.add((i, j))  # Mark current cell as visited

found = (dfs(board, word[1:], visited.copy(), i-1, j) or  # Explore up
         dfs(board, word[1:], visited.copy(), i+1, j) or  # Explore down
         dfs(board, word[1:], visited.copy(), i, j-1) or  # Explore left
         dfs(board, word[1:], visited.copy(), i, j+1))  # Explore right

visited.remove((i, j))


# maybe if we are thinking about the possibilities we need to do + or - or something else identifying whether it is 0/1 knapsack
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        dp={} #cache
        def dfs(i,total):
            if i==len(nums):
                if total==target:
                    return 1
                else:
                    return 0
            if (i,total) in dp:
                return dp[(i,total)]
            dp[(i,total)]=dfs(i+1,total+nums[i])+dfs(i+1,total-nums[i]) # adding + because we need both possibilities
            return dp[(i,total)]
        return dfs(0,0)
# first lets identify whether it is a 0/1 knapsack or something else
# if it is like find longest or minimum then use something like 1+dfs()


class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        dp={}
        def helper(index,targetSum,s):
            if (index,s) in dp:
                return dp[(index,s)]
            if index>len(nums)-1:
                return False
            if s==targetSum:
                return True
            if s>targetSum:
                return False
            dp[(index,s)]=helper(index+1,targetSum,s+nums[index]) or helper(index+1,targetSum,s)
            return dp[(index,s)]
        if sum(nums)%2!=0:
            return False
        return helper(0,sum(nums)//2,0)


class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        directions=[(0,1),(1,0),(-1,0),(0,-1)]
        self.t=float('-inf')
        def path(i,j,total):
            self.t=max(self.t,total)
            if i<0 or j<0 or i>len(grid)-1 or j>len(grid[0])-1:
                return 0
            if (i,j) in visited:
                return 1
            if grid[i][j]==0:
                return 0
            visited.add((i,j))
            max_path=float('-inf')
            for k,l in directions:
                path(i+k,j+l,total+grid[i][j])
                # Re-exploring Cells in Different Paths: By removing the cell from visited after exploring all paths starting from it, the code allows for revisiting the same cell in different paths. For example, consider a cell with valid paths going both up and down. If visited is not removed, the function would only explore one direction, missing the other.
            visited.remove((i,j))
            return self.t
        visited=set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                path(i,j,0)
                visited=set()
        return self.t
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        @cache
        def targetSum(i,total):
            if i==len(nums) and total==target:
                return 1
            elif i>=len(nums):
                return 0
            k=targetSum(i+1,total+nums[i])+targetSum(i+1,total-nums[i])  #unique way fo doing it remember if it is including all the values but different symbols like [-1,1,-1,1,1] or [1,1,1,-1,-1] to the target sum nice
            return k
        return targetSum(0,0)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        def dfs(p,q):
            if p or q:
                if (p and not q) or (q and not p):
                    return False
                if p.val==q.val:
                    return dfs(p.left,q.left) and dfs(p.right,q.right)
                else:
                    return False
            else:
                return True
        return dfs(p,q)


#important in finding max safest path may15th leetcode
# Grid Initialization and Multi-source BFS:

# Initial grid:
# 1 0 0
# 0 0 0
# 0 0 1
# Distance Calculation (Multi-source BFS):

# Distance grid after BFS:
# 0 1 2
# 1 2 1
# 2 1 0
# [[3, 2, 1, 0], 
#  [2, 3, 2, 1], 
#  [1, 2, 3, 2], 
#  [0, 1, 2, 3]]

#finding subarrays sum =k or not we should always plus the prefizsum and check prefixsum-k in d and remainig you will know
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefixsum=c=0
        d=dict()
        d[0]=1 #For the condition when prefixsum = k
        for i in nums:
            prefixsum+=i
            if prefixsum-k in d:
                c+=d[prefixsum-k]
            if prefixsum in d:
                d[prefixsum]+=1
            else:
                d[prefixsum]=1
        return c                
  

class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        dp = [-1 for _ in range(n+1)]
        def recursion(index):
            if(index == n):
                return 1
            if(s[index] == '0'):
                return 0
            if(index == n-1 ):
                return 1
            if dp[index] != -1:
                return dp[index]
            ans1 = recursion(index+1) # we can also do something like this where here in this case formming soemthign like string='123' 1-->A,2-->B how can we divide like that i mean in how many ways
            if(int(s[index:index+2]) < 27):
                ans1 += recursion(index+2)
            dp[index] = ans1
            return ans1
        return recursion(0)

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        directions=[(1,0),(0,1)]
        dp={}
        def uniquePaths(i,j,visited):
            if (i,j) in dp:
                return dp[i,j]
            if i==m-1 and j==n-1:
                return 1
            if i>=m or j>=n:
                return 0
            k=0
            for r,c in directions:
                if 0<=i+r<m and 0<=j+c<n and (i+r,j+c) not in visited:
                    visited.add((i+r,j+c)) # this is very important we are adding visited like this and removing afterthat because (0,0) already exists and it will satrt from there
                    k+=uniquePaths(i+r,j+c,visited)
                    visited.remove((i+r,j+c))
            dp[(i,j)]=k
            return k
        
        return uniquePaths(0,0,set())


class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        insertIndex=bisect_left(intervals,newInterval) # bisect left will find the index where we can insert and there we are inserting time complexity is logn
        intervals.insert(insertIndex,newInterval)
        stack=[intervals[0]]
        for i in range(1,len(intervals)):
            if stack[-1][0]<=intervals[i][0]<=stack[-1][1]:
                stack[-1][0]=stack[-1][0]
                stack[-1][1]=max(stack[-1][1],intervals[i][1])
            else:
                stack.append(intervals[i])
        return stack


intervals=sorted(intervals,key=lambda x:x[0]) # just for reference like how to use key and lamda
            
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        count=0
        prefix_sum=0
        prefix_map={0:1}
        # here first we are checking whether that reminder existed or not if it exist if merge those numbers we get reminder 0 thats logic ,mainly if remainder already increase increase the map and count says these many subarrys aexisted
        for i in nums:
            prefix_sum+=i
            mod=prefix_sum%k
            if mod in prefix_map:
                count+=prefix_map[mod]
                prefix_map[mod]+=1
            else:
                prefix_map[mod]=1
        return count

