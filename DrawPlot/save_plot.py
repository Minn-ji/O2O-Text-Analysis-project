import os
import matplotlib.pyplot as plt

# 마지막 출력 이미지를 저장

def save_plot(filename: str, folder: str = "output"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    plt.savefig(path, bbox_inches='tight', dpi=300)