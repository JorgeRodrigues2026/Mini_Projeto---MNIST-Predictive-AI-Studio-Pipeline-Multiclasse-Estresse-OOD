"""
Módulo de pré-processamento de imagens digitais e cálculo de centro de massa.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift


def normalize_pixels(X):
    """
    Normaliza os valores de intensidade de pixel da escala [0, 255] para [0.0, 1.0].
    Essencial para convergência numérica de otimizadores baseados em gradiente e distâncias euclidianas.
    """
    return X / 255.0


def preprocess_custom_image(image_input, target_size=(28, 28)):
    """
    Pipeline de pré-processamento para dígitos manuscritos externos (OpenCV/PIL):
    1. Conversão para escala de cinza (Luminância)
    2. Detecção automática e inversão de cores (papel claro com traço escuro -> fundo preto com traço branco)
    3. Recorte pela caixa delimitadora (Bounding Box)
    4. Redimensionamento mantendo proporção com preenchimento em 20x20
    5. Centralização da massa do traço na grade 28x28
    6. Normalização para [0.0, 1.0]
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('L')
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            img = Image.fromarray(image_input).convert('L')
        else:
            img = Image.fromarray(image_input, mode='L')
    else:
        img = image_input.convert('L')

    img_arr = np.array(img, dtype=np.float32)

    # Inversão: Se a média dos cantos for clara (> 127), inverte para traço branco em fundo preto
    corners = [img_arr[0, 0], img_arr[0, -1], img_arr[-1, 0], img_arr[-1, -1]]
    if np.mean(corners) > 127:
        img_arr = 255.0 - img_arr

    # Limiarização suave para remover ruído de fundo
    img_arr[img_arr < 40] = 0.0

    # Bounding Box dos pixels com traço
    rows = np.any(img_arr > 50, axis=1)
    cols = np.any(img_arr > 50, axis=0)
    if not np.any(rows) or not np.any(cols):
        # Imagem em branco
        return np.zeros((1, 784), dtype=np.float32), np.zeros((28, 28), dtype=np.float32)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = img_arr[rmin:rmax + 1, cmin:cmax + 1]

    # Redimensiona para caber em uma caixa de 20x20 mantendo proporção (padrão MNIST)
    crop_img = Image.fromarray(cropped.astype(np.uint8))
    w, h = crop_img.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))
    else:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))

    resized = crop_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(resized, dtype=np.float32)

    # Insere em uma tela preta 28x28 centralizada inicialmente
    padded_28 = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    padded_28[start_y:start_y + new_h, start_x:start_x + new_w] = resized_arr

    # Centralização fina por Centro de Massa (Center of Mass Shift)
    cy, cx = center_of_mass(padded_28)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_y = int(np.round(14.0 - cy))
        shift_x = int(np.round(14.0 - cx))
        # Limita o deslocamento para evitar translações bruscas
        shift_y = np.clip(shift_y, -4, 4)
        shift_x = np.clip(shift_x, -4, 4)
        padded_28 = shift(padded_28, shift=(shift_y, shift_x), mode='constant', cval=0.0)

    # Normalização final
    normalized_28 = padded_28 / 255.0
    vector_784 = normalized_28.reshape(1, 784)

    return vector_784, normalized_28