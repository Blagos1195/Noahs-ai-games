import torch

class TensorReplayBuffer:
    def __init__(self, capacity, device, rows=6, columns=7):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((capacity, 2, rows, columns), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, 2, rows, columns), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.float32, device=device)

    def push_batch(self, states, actions, rewards, next_states, dones):
        batch_size = states.size(0)
        
        if batch_size > self.capacity:
            states = states[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            next_states = next_states[-self.capacity:]
            dones = dones[-self.capacity:]
            batch_size = self.capacity

        indices = (torch.arange(self.ptr, self.ptr + batch_size, device=self.device) % self.capacity).long()

        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.dones[indices] = dones

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )