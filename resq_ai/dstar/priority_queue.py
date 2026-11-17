"""
resq_ai.dstar.priority_queue
============================
Hash-indexed binary min-heap for D* Lite key pairs (k1, k2).
Guarantees O(log N) insert, update, remove, and extract_min,
and O(1) membership check and top_key inspection.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any


KeyType = Tuple[float, float]


class IndexedPriorityQueue:
    """
    Binary min-heap where keys are lexicographical tuples (k1, k2)
    and items have O(1) position lookup.
    """

    def __init__(self):
        # Array of [key, item]
        self.heap: List[List[Any]] = []
        # Mapping from item to index in self.heap
        self.pos: Dict[Any, int] = {}

    def __len__(self) -> int:
        return len(self.heap)

    def is_empty(self) -> bool:
        return len(self.heap) == 0

    def contains(self, item: Any) -> bool:
        return item in self.pos

    def top_key(self) -> KeyType:
        """Returns the minimum key (k1, k2) without popping, or (inf, inf) if empty."""
        if not self.heap:
            return (float("inf"), float("inf"))
        return self.heap[0][0]

    def insert(self, item: Any, key: KeyType) -> None:
        """Inserts item with key (k1, k2). Raises ValueError if already present."""
        if item in self.pos:
            self.update(item, key)
            return

        index = len(self.heap)
        entry = [key, item]
        self.heap.append(entry)
        self.pos[item] = index
        self._bubble_up(index)

    def update(self, item: Any, new_key: KeyType) -> None:
        """Updates the priority key of an existing item in O(log N)."""
        if item not in self.pos:
            self.insert(item, new_key)
            return

        idx = self.pos[item]
        old_key = self.heap[idx][0]
        self.heap[idx][0] = new_key

        if new_key < old_key:
            self._bubble_up(idx)
        elif new_key > old_key:
            self._bubble_down(idx)

    def remove(self, item: Any) -> None:
        """Removes arbitrary item from heap in O(log N)."""
        if item not in self.pos:
            return

        idx = self.pos[item]
        last_idx = len(self.heap) - 1

        if idx == last_idx:
            del self.pos[item]
            self.heap.pop()
            return

        # Swap with last element
        self._swap(idx, last_idx)
        del self.pos[item]
        self.heap.pop()

        # Restore heap invariant
        self._bubble_down(idx)
        self._bubble_up(idx)

    def pop_min(self) -> Tuple[Any, KeyType]:
        """Extracts and returns the (item, key) with the smallest key."""
        if not self.heap:
            raise IndexError("pop_min from empty priority queue")

        min_entry = self.heap[0]
        item = min_entry[1]
        key = min_entry[0]

        last_idx = len(self.heap) - 1
        if last_idx == 0:
            del self.pos[item]
            self.heap.pop()
            return item, key

        # Swap root with last element
        self._swap(0, last_idx)
        del self.pos[item]
        self.heap.pop()

        self._bubble_down(0)
        return item, key

    def _swap(self, i: int, j: int) -> None:
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.pos[self.heap[i][1]] = i
        self.pos[self.heap[j][1]] = j

    def _bubble_up(self, idx: int) -> None:
        while idx > 0:
            parent = (idx - 1) // 2
            # Compare keys: heap[idx][0] < heap[parent][0]
            if self.heap[idx][0] < self.heap[parent][0]:
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _bubble_down(self, idx: int) -> None:
        size = len(self.heap)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx

            if left < size and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < size and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right

            if smallest != idx:
                self._swap(idx, smallest)
                idx = smallest
            else:
                break
