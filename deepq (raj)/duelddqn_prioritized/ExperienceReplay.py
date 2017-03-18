import numpy as np
import random

from collections import namedtuple

class Replay():
	def __init__(self, capacity):
		self.replay_memory = SumTree(capacity)
		self.transition = namedtuple("transition", ["state","action","reward","next_state","done"])

	def add(self, error,state,action,reward,next_state,done):
		self.replay_memory.add(error, self.transition(state,action,reward,next_state,done))

	def sample(self, n):
		batch = []
		segment = self.replay_memory.total() / n

		for i in range(n):
			a = segment * i
			b = segment * (i + 1)

			s = random.uniform(a, b)
			(idx, p, data) = self.replay_memory.get(s)
			batch.append((idx, data))

		return batch

	def update(self, idx, error):
		self.replay_memory.update(idx, error)

class SumTree():
	write = 0

	def __init__(self, capacity):
		self.capacity = capacity
		self.tree = np.zeros(2*capacity - 1)
		self.data = np.zeros(capacity, dtype = object)

	def propagate(self, idx, change):
		parent = (idx - 1) // 2

		self.tree[parent] += change

		if parent != 0:
			self.propagate(parent, change)

	def retrieve(self, idx, s):
		left = 2 * idx + 1
		right = left + 1

		if left >= len(self.tree):
			return idx

		if s <= self.tree[left]:
			return self.retrieve(left, s)
		else:
			return self.retrieve(right, s-self.tree[left])

	def total(self):
		return self.tree[0]

	def add(self, p, data):
		idx = self.write + self.capacity - 1

		self.data[self.write] = data
		self.update(idx, p)

		self.write += 1
		if self.write >= self.capacity:
			self.write = 0

	def update(self, idx, p):
		change = p - self.tree[idx]

		self.tree[idx] = p
		self.propagate(idx, change)

	def get(self,s):
		idx = self.retrieve(0, s)
		dataIdx = idx - self.capacity + 1

		return (idx, self.tree[idx], self.data[dataIdx])