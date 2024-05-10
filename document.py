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
        
        
  

