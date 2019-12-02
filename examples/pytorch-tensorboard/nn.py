import torch
import torchvision

lr = 0.001
batch_size = 256
hidden_size = 100
epochs = 3


def get_loader(train):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "/tmp/mnist", train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x.view(x.shape[0], -1)[0,:])
            ])),
        batch_size=batch_size, shuffle=True
    )

train_loader = get_loader(True)
test_loader = get_loader(False)


net = torch.nn.Sequential(
    torch.nn.BatchNorm1d(784),
    torch.nn.Linear(784, hidden_size),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(hidden_size, 10),
    torch.nn.Softmax(dim=1)
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


for epoch in range(epochs):
    for x, y in train_loader:
        out = net(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        acc = (out.max(dim=1).indices == y).float().sum()/x.shape[0]

        print("epoch:%d  accuracy:%.2f" % (epoch, acc.item()))

rights = 0
totals = 0
for x, y in test_loader:
    out = net(x)
    rights += (out.max(dim=1).indices == y).float().sum().item()
    totals += x.shape[0]

print("Accuracy: %.4f" % (rights/totals))