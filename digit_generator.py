"""
Módulo gerador de imagens 28x28 PNG de dígitos manuscritos e sintéticos para testes.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_and_save_digits_png(output_dir="data/custom_digits"):
    """
    Cria 10 imagens no formato .PNG de tamanho 28x28 com dígitos de 0 a 9.
    Gera as imagens com fundo branco e traço preto espesso, simulando escrita real.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for digit in range(10):
        # Cria imagem de alta resolução (140x140) para rasterização precisa com anti-aliasing
        high_res = Image.new('L', (140, 140), color=255)
        draw = ImageDraw.Draw(high_res)

        # Desenha formas manuscritas com espessura
        if digit == 0:
            draw.ellipse([30, 20, 110, 120], outline=0, width=16)
        elif digit == 1:
            draw.line([(70, 25), (70, 115)], fill=0, width=16)
            draw.line([(45, 50), (70, 25)], fill=0, width=14)
            draw.line([(40, 115), (100, 115)], fill=0, width=14)
        elif digit == 2:
            draw.arc([30, 20, 110, 75], start=180, end=0, fill=0, width=16)
            draw.line([(110, 48), (30, 115)], fill=0, width=16)
            draw.line([(30, 115), (110, 115)], fill=0, width=16)
        elif digit == 3:
            draw.arc([35, 20, 105, 70], start=210, end=30, fill=0, width=15)
            draw.arc([35, 65, 105, 118], start=330, end=150, fill=0, width=15)
        elif digit == 4:
            draw.line([(95, 20), (95, 120)], fill=0, width=16)
            draw.line([(95, 20), (30, 85)], fill=0, width=15)
            draw.line([(25, 85), (115, 85)], fill=0, width=16)
        elif digit == 5:
            draw.line([(105, 25), (35, 25)], fill=0, width=16)
            draw.line([(35, 25), (35, 65)], fill=0, width=16)
            draw.arc([35, 55, 105, 118], start=270, end=110, fill=0, width=16)
        elif digit == 6:
            draw.arc([35, 20, 105, 115], start=45, end=270, fill=0, width=16)
            draw.ellipse([35, 60, 105, 118], outline=0, width=16)
        elif digit == 7:
            draw.line([(30, 25), (110, 25)], fill=0, width=16)
            draw.line([(110, 25), (45, 118)], fill=0, width=16)
            draw.line([(55, 70), (90, 70)], fill=0, width=14)
        elif digit == 8:
            draw.ellipse([40, 20, 100, 68], outline=0, width=15)
            draw.ellipse([35, 62, 105, 120], outline=0, width=15)
        elif digit == 9:
            draw.ellipse([35, 20, 105, 78], outline=0, width=16)
            draw.arc([35, 25, 105, 120], start=270, end=135, fill=0, width=16)

        # Redimensiona para o formato padrão 28x28 com interpolação Lanczos
        img_28 = high_res.resize((28, 28), Image.Resampling.LANCZOS)
        
        file_path = os.path.join(output_dir, f"digit_{digit}.png")
        img_28.save(file_path, format="PNG")
        saved_files.append(file_path)

    print(f"10 imagens PNG (28x28) geradas com sucesso na pasta: {output_dir}")
    return saved_files