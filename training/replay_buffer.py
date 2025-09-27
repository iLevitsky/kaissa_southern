# training/replay_buffer.py

class ReplayBuffer:
    """
    In-memory buffer for small-scale usage. 
    If you're storing everything as JSON, you might not need this.
    """
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []

    def push(self, state_tensor, policy_vec, value):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state_tensor, policy_vec, value))

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch

    def __len__(self):
        return len(self.buffer)
