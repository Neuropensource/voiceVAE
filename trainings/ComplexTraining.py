def LRcompute():
    return 0.001

for epoch in range(nb_epochs):  # loop over the dataset multiple times
    optimizer = optim.SGD(net.parameters(), lr=LRcompute(), momentum=0.9)
    make_static(parameters)
    for batch in dataloader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()