# 机考复习

## 排序，队列，栈

中序表达式转后序

```python
def infix_to_postfix(expression):
	pre = {'+':1, '-':1, '*':2, '/':2}
	stack = []
	postfix = []
	number = ''
	for char in expression:
	if char.isnumeric() or char == '.':
 		number += char
 	else:
 		if number:
 			num = float(number)
 			postfix.append(int(num) if num.is_integer() else num)
 			number = ''
 		if char in '+-*/':
 			while stack and stack[-1] in '+-*/' and precedence[char]<=precedence[stack[-1]]:
 				postfix.append(stack.pop())
 			stack.append(char)
 		elif char == '(':
 			stack.append(char)
 		elif char == ')':
 			while stack and stack[-1] != '(':
 				postfix.append(stack.pop())
 			stack.pop()
 	if number:
 		num = float(number)
 		postfix.append(int(num) if num.is_integer() else num)
 	while stack:
 		postfix.append(stack.pop())
 	return ' '.join(str(x) for x in postfix)
```

合法出栈序列

```python
def isPopSeq(s1,s2):
	stack=[]
	if len(s1)!=len(s2):
		return False
	else:
		n=len(s1)
		stack.append(s1[0])
		k1,k2=1,0
		while k1<n:
			if len(stack)>0 and stack[-1]==s2[k2]:
				stack.pop()
				k2+=1
			else:
				stack.append(s1[k1])
				k1+=1
		return ''.join(stack[::-1])==s2[k2:]
```

 单调栈（模板

```python
n = int(input()) 
a = list(map(int, input().split())) 
stack = [] 
for i in range(n):
	while stack and a[stack[-1]] < a[i]:
 		a[stack.pop()] = i + 1
	stack.append(i)
while stack:
	a[stack[-1]] = 0
	stack.pop()
print(*a)
```

## 类

```python
#创建一个类
class person():
	#类变量的创建
	name='aa'
	#类方法的创建
	def who(self):
		print(name)

#类变量的访问：类名.变量名
p=person.name
print(p)

#实例化类（类函数的使用）：类名.函数名
c=person()
c.who()
#输出：aa

#类变量的修改:实例化后只修改自己内部而不影响原始的类；若直接用类名修改则会影响所有的实例化
c=person()
c.name=0
print(c.name)
#0
a=person()
print(a.name)
#aa
person.name=1
#则后续所有都会改变

#构造器：
class person():
	#self 表示的就是类的实例
	def __init__(self,a)
	#实例变量：依靠输入
	self.name=a
	#实例变量：默认值
	self.age=10
print(person(a).name)
#a


#用类实现双端队列：
class deque:
	def __init__(self):
		self.queue=[]
	
    def push(self,a):#进队
		self.queue.append(a)

	def post_out(self):#队尾出队
		self.queue.pop()

    def pre_out(self):#队头出队
		self.queue.pop(0)

    def empty(self):#判断是否为空
		if self.queue==[]:
			return False
		else:
			return True
```

## 树

### 前中序转后序，中后序转前序

```python
def postorder(preorder,inorder):
    if not preorder:
        return ''
    root=preorder[0]
    idx=inorder.index(root)
    left=postorder(preorder[1:idx+1],inorder[:idx])
    right=postorder(preorder[idx+1:],inorder[idx+1:])
    return left+right+root
def preorder(inorder,postorder):
    if not inorder:
        return ''
    root=postorder[-1]
    idx=inorder.index(root)
    left=preorder(inorder[:idx],postorder[:idx])
    right=preorder(inorder[idx+1:],postorder[idx:-1])
    return root+left+right
```

### 前，中，后序遍历，层次遍历

```python
def preorder_traversal(root):
    result = []
    def traverse(node):
        if node:
            result.append(node.val)  
            traverse(node.left)     
            traverse(node.right)     
    traverse(root)
    return result


def inorder_traversal(root):
    result = []
    def traverse(node):
        if node:
            traverse(node.left)     
            result.append(node.val)  
            traverse(node.right)     
    traverse(root)py
    return result


def postorder_traversal(root):
    result = []
    def traverse(node):
        if node:
            traverse(node.left)      
            traverse(node.right)     
            result.append(node.val)  
    traverse(root)
    return result


from collections import deque
def levelorder(root):
    if not root:
        return ""
    queue=deque([root])
    result=""
    while queue:
        node=queue.popleft()
        result+=node.val
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result
```

### 二叉搜索树

```python
def insert(root,num):
    if not root:
        return Node(num)
    if num<root.val:
        root.left=insert(root.left,num)
    else:
        root.right=insert(root.right,num)
    return root
```

### 字典树的构建

```python
def insert(root,num):
	node=root
	for digit in num:
		if digit not in node.children:
			node.children[digit]=TrieNode()
		node=node.children[digit]
		node.cnt+=1
```

### 并查集

```python
class disj_set:
    def __init__(self,n):
        self.rank = [1 for i in range(n)]
        self.parent = [i for i in range(n)]

    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
    def union(self,x,y):
        x_root = self.find(x)
        y_root = self.find(y)
        
        if x_root == y_root:
            return
        
        if self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        
        elif self.rank[y_root] < self.rank[x_root]:
            self.parent[x_root] = y_root
        
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1
            
count = 0
for x in range(1,n+1):
    if D.parent[x-1] == x - 1:
        count += 1
```



## 图

### 判断无向图是否连通，有回路

```python
class Node:
    def __init__(self, v):
        self.value = v
        self.joint = set()
def connected(x, visited, num):#判断是否联通
    visited.add(x)
    al = 1
    q = [x]
    while al != num and q:
        x = q.pop(0)
        for y in x.joint:
            if y not in visited:
                visited.add(y)
                al += 1
                q.append(y)
    return al == num
def loop(x, visited, parent):#判断是否有环
    visited.add(x)
    if x.joint:
        for a in x.joint:
            if a in visited and a != parent:
                return True
            elif a != parent and loop(a, visited, x):
                return True
    return False
n, m = map(int, input().split())
vertex = [Node(i) for i in range(n)]
for i in range(m):
    a, b = map(int, input().split())
    vertex[a].joint.add(vertex[b])
    vertex[b].joint.add(vertex[a])
if connected(vertex[0], set(), n):
    print('connected:yes')
else:
    print('connected:no')
x=0
for i in range(n):
    if loop(vertex[i],set(),None):
        print('loop:yes')
        x=1
        break
if x==0:
print('loop:no')
```

### 拓扑排序：判断有向图是否有环

```python
from collections import deque
class Node:
    def __init__(self, v):
        self.val = v
        self.to = []

t = int(input())
for _ in range(t):
    n, m = map(int, input().split())
    node = [Node(i) for i in range(1, n + 1)]
    into = [0 for _ in range(n)] 
    for _ in range(m): 
        x, y = map(int, input().split())
        node[x - 1].to.append(node[y - 1])
        into[y - 1] += 1
    queue = deque([node[i] for i in range(n) if into[i] == 0])
    output = []
    
    while queue:
        a = queue.popleft()
        output.append(a)
        for x in a.to:
            num = x.val
            into[num - 1] -= 1
            if into[num - 1] == 0:
                queue.append(x)
    if len(output) == n:
        print('No')
    else:
        print('Yes')
```

### bfs

```python 
from collections import deque
def bfs(graph, start_node):
    queue = deque([start_node])
    visited = set()
    visited.add(start_node)
    while queue:
        current_node = queue.popleft()
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### dijkstra：最短路径

```python
def dijkstra(start,end):
    heap=[(0,start,[start])]
    vis=set()
    while heap:
        (cost,u,path)=heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u==end: return (cost,path)
        for v in graph[u]:
            if v not in vis:
            	heappush(heap,(cost+graph[u][v],v,path+[v]))

import heapq
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances
```

### 强连通分量Kosaraju

```python
def dfs1(graph, node, visited, stack):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs1(graph, neighbor, visited, stack)
    stack.append(node)
def dfs2(graph, node, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs2(graph, neighbor, visited, component)
def kosaraju(graph):
    stack = []
    visited = [False] * len(graph)
    for node in range(len(graph)):
        if not visited[node]:
            dfs1(graph, node, visited, stack)
    transposed_graph = [[] for _ in range(len(graph))]
    for node in range(len(graph)):#取反
        for neighbor in graph[node]:
            transposed_graph[neighbor].append(node)
    visited = [False] * len(graph)
    results = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            result = []
            dfs2(transposed_graph, node, visited,result)
            rlsults.append(result)
    return results
```



### 最小生成树

#### kruskal

```python
class DisjointSet:
    def __init__(self, num_vertices):
        self.parent = list(range(num_vertices))
        self.rank = [0] * num_vertices
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1
def kruskal(graph):
    num_vertices = len(graph)
    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))
                
    edges.sort(key=lambda x: x[2])
    disjoint_set = DisjointSet(num_vertices)
    minimum_spanning_tree = []
    for edge in edges:
        u, v, weight = edge
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            minimum_spanning_tree.append((u, v, weight))
    return minimum_spanning_tree
```

#### prim

```python
def prim(graph, start):
    pq = []
    start.distance = 0
    heapq.heappush(pq, (0, start))
    visited = set()
    while pq:
        currentDist, currentVert = heapq.heappop(pq)
        if currentVert in visited:
            continue
        visited.add(currentVert)
        for nextVert in currentVert.getConnections():
            weight = currentVert.getWeight(nextVert)
            if nextVert not in visited and weight < nextVert.distance:
                nextVert.distance = weight
                nextVert.pred = currentVert
                heapq.heappush(pq, (weight, nextVert))
```