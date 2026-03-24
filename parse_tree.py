import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import deque

'''
JSON file
    ↓
parse_tree_json()
    ↓
TreeNode objects (tree structure)
    ↓
iter_nodes()  → traversal
    ↓
build_keyword_index() → fast lookup
    ↓
query_with_index() → search
    ↓
pretty_print_nodes() → output

'''

# creating perticular node for search
@dataclass
class TreeNode:
    keyword: str
    depth: int
    response: str
    parent: Optional["TreeNode"] = field(default=None, repr=False)
    children: List["TreeNode"] = field(default_factory=list)

    @property
    def clean_response(self) -> str:
        text = self.response.strip()
        text = re.sub(r"(?i)^definition:\s*", "", text)
        return text

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self):
        return f"TreeNode(keyword={self.keyword!r}, depth={self.depth}, children={len(self.children)})"



# Actual Json Tree Parser 
def _parse_node(keyword: str, node_dict: dict, parent: Optional[TreeNode] = None) -> TreeNode:
    '''
        Creates a TreeNode
        Reads children from JSON
        Recursively parses children
        Links parent ↔ child
        Returns node
    '''
    node = TreeNode(
        keyword=keyword,
        depth=node_dict["depth"],
        response=node_dict.get("response", ""),
        parent=parent,
    )

    for child_kw, child_dict in node_dict.get("children", {}).items():
        child = _parse_node(child_kw, child_dict, parent=node)
        node.children.append(child)

    return node


def parse_tree_json(path: str) -> TreeNode:

    '''
        Opens JSON file
        Loads JSON
        Gets root node
        Calls recursive parser
        Returns root TreeNode
    '''

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) != 1:
        raise ValueError(f"Expected exactly 1 root key in {path}")

    root_keyword, root_dict = next(iter(data.items()))
    return _parse_node(root_keyword, root_dict)


# BFS traversal 
def iter_nodes(root: TreeNode):
    """Breadth-first traversal"""
    queue = deque([root])
    while queue:
        node = queue.popleft()
        yield node
        queue.extend(node.children)



# search node - Query hit 
def find_nodes_info(root: TreeNode, keyword: str):
    '''
        For each matching node it returns:
            node_name
            parent_name
            children
            depth
            path
            response
    '''
    keyword_lower = keyword.lower()
    results = []

    for node in iter_nodes(root):
        if node.keyword.lower() == keyword_lower:
            results.append({
                "node_name": node.keyword,
                "parent_name": node.parent.keyword if node.parent else None,
                "children": [child.keyword for child in node.children],
                "depth": node.depth,
                "path": " > ".join(get_path_to_root(node)),
                "response": node.clean_response
            })

    return results


def get_path_to_root(node: TreeNode) -> List[str]:
    path = []
    current = node
    while current is not None:
        path.append(current.keyword)
        current = current.parent
    return list(reversed(path))



# Index - Fast Lookups 
def build_keyword_index(root: TreeNode) -> Dict[str, List[TreeNode]]:

    '''
        Builds dictionary - to reduce searching time from O(N) to O(1)
        {
            "governance": [node1, node2],
            "rules": [node3],
            "policy": [node4]
        }
    '''
    index = {}
    for node in iter_nodes(root):
        key = node.keyword.lower()
        index.setdefault(key, []).append(node)
    return index


def query_with_index(index, keyword: str):
    '''
        uses index to find the node in dictionary
    '''
    nodes = index.get(keyword.lower(), [])
    results = []

    for node in nodes:
        results.append({
            "node_name": node.keyword,
            "parent_name": node.parent.keyword if node.parent else None,
            "children": [child.keyword for child in node.children],
            "depth": node.depth,
            "path": " > ".join(get_path_to_root(node)),
            "response": node.clean_response
        })

    return results

# pretty printing for our need 
def pretty_print_nodes(results):
    if not results:
        print("No nodes found")
        return

    for i, r in enumerate(results, 1):
        print(f"\nMatch {i}")
        print("-" * 50)
        print(f"Node     : {r['node_name']}")
        print(f"Parent   : {r['parent_name']}")
        print(f"Depth    : {r['depth']}")
        print(f"Path     : {r['path']}")
        print(f"Children : {', '.join(r['children']) if r['children'] else 'None'}")
        print(f"Response : {r['response']}")


# find problematic nodes 

# load tree 
def load_tree(file_path: str) -> TreeNode:
    return parse_tree_json(file_path)

# find desired keyword in the tree
def query_tree(file_path: str, keyword: str) -> List[Dict]:
    root = parse_tree_json(file_path)
    return find_nodes_info(root, keyword)


# test the tree parser 

root = load_tree("./json_trees_for_testing/tree_depends.json")
index = build_keyword_index(root)  # traverses a tree and build dictionary of each keyword
# print(index)
results = query_with_index(index, "circumstance") # hit the dictionary and find all the occurance of the word
pretty_print_nodes(results)