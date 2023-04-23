import logging

logging.basicConfig(filename='training.log', level=logging.DEBUG)

def train(model, optimizer, train_loader, device):
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward pass
            scores = model(data)
            loss = F.cross_entropy(scores, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient descent
            optimizer.step()

            # Log loss
            if batch_idx % LOG_FREQUENCY == 0:
                step = epoch * len(train_loader) + batch_idx
                logging.debug(f"Loss at step {step}: {loss}")
