from torch.utils.tensorboard import SummaryWriter

def log_to_tensorboard(writer, loss, accuracy, epoch):
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
