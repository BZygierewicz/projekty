from mnist_model import *
from GUI_app import *

# Wczytanie wcześniej wytrenowanego modelu
model.load_state_dict(torch.load("better_model.pth", map_location=device))  # Wczytaj model
model.to(device)
model.eval()

if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()