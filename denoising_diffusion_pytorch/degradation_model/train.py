from dataloader import *
from model import *

data_path = "/home/share/CARE/Isotropic_Liver/train_data/data_label.npz"
train_loader, val_loader = get_Dataloader(data_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=1, out_channels=1)
model.to(device,dtype=torch.float32)

optimizer = get_optimiazer(model, learning_rate=0.001)
loss_fn = get_loss()
num_epochs = 200
save_path = './checkpoint/best.pth'

trainer(train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=num_epochs,
        device=device,
        save_path=save_path)
