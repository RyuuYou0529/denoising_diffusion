import torch
import torch.nn as nn
from torch import optim

from tqdm import tqdm

# ============================
# Model
# ============================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features[:-1]):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            if i != len(self.downs)-1:
                x = nn.MaxPool2d(kernel_size=2)(x)

        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i//2+1]
            if x.shape != skip.shape:
                x = nn.functional.pad(x, (0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]))
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)

        x = self.final_conv(x)
        return x

# ============================
# Loss
# ============================
def get_loss():
    return nn.MSELoss()

def get_optimiazer(model, learning_rate=0.001):
    return optim.Adam(model.parameters(), lr=learning_rate)


# ============================
# Trainer
# ============================

def trainer(train_loader, val_loader, model, optimizer, loss_fn, num_epochs, device, save_path: str):
    if not save_path.endswith('pth'):
        raise Exception('Invalid save path')
    
    best_loss = 1
    for epoch in range(num_epochs):
        # =======================
        # train one epoch
        # =======================
        model.train()
        train_loop = tqdm(enumerate(train_loader), ncols=100, total=len(train_loader))
        train_loop.set_description(f'[Train Epoch] [{epoch+1}/{num_epochs}]')
        for i, (LR, HR) in train_loop:
            LR = LR.to(device)
            HR = HR.to(device)

            # forward
            optimizer.zero_grad()
            outputs = model(HR)
            loss = loss_fn(outputs, LR)

            # backward
            loss.backward()
            optimizer.step()

            train_loop.set_postfix_str(f'loss={loss.item():.6f}')
        # =======================

        # =======================
        # val one epoch
        # =======================
        model.eval()
        total_loss = 0
        val_loop = tqdm(enumerate(val_loader), ncols=100, total=len(val_loader))
        val_loop.set_description(f'[Val Epoch] [{epoch+1}/{num_epochs}]')

        with torch.no_grad():
            for i, (LR, HR) in val_loop:
                LR = LR.to(device)
                HR = HR.to(device)

                # forward
                optimizer.zero_grad()
                outputs = model(HR)
                loss = loss_fn(outputs, LR)
                
                total_loss += loss.item()

                val_loop.set_postfix_str(f'loss={loss.item():.6f}')

        total_loss /= len(val_loader)
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), save_path)
            print(f'Save checkpoint in Epoch[{epoch+1}/{num_epochs}].')