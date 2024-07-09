import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.layer = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))

def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('/p/scratch/deepacf/vasireddy1/MNIST/', train=True, download=False, transform=transform)
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(mnist_train, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(mnist_val, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(mnist_train, batch_size=64, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(mnist_val, batch_size=64, sampler=val_sampler, num_workers=4)

    model = MNISTModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    for epoch in range(10):
        # Training phase
        # TODO: Implement the training phase during training HERE

        # Validation phase
        ddp_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(rank), y.to(rank)
                outputs = ddp_model(x)
                loss = F.cross_entropy(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total

        print(f"Rank {rank}, Epoch {epoch}, Loss {val_loss}, Accuracy {accuracy}")

    dist.destroy_process_group()

def main():
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
