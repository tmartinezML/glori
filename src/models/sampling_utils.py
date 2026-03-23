import math
import itertools

import torch
from PIL import Image, ImageDraw, ImageFont


def auto_fontsize(text, image_size=(64, 64), margin=0.05):
    """
    Calculate font size directly from character metrics without test rendering.

    Parameters:
    -----------
    text : str
        Text to render
    target_area : float
        Target area in pixels²
    image_size : tuple
        Image dimensions
    font_path : str
        Font file path

    Returns:
    --------
    int : Calculated font size
    """
    # Get text dimensions - count actual characters
    lines = text.split("\n")
    n_lines = len(lines)
    max_chars_per_line = max(len(line) for line in lines)

    # For monospace fonts, we can predict dimensions directly
    # Typical character metrics for monospace fonts:
    char_width_ratio = 0.6  # Character width ≈ 0.6 * font_size
    char_height_ratio = 0.8  # Character height ≈ 0.8 * font_size
    line_spacing = 0.2  # Line spacing = int(0.2 * font_size)

    # See whether height or width is the limiting factor
    if (n_lines * (char_height_ratio + line_spacing)) / (
        max_chars_per_line * char_width_ratio
    ) > (image_size[1] / image_size[0]):
        # Height is limiting factor. Choose font size based on image height
        font_size = (
            (1 - 2 * margin)
            * image_size[1]
            / (n_lines * (char_height_ratio + line_spacing))
        )
    else:
        # Width is limiting factor. Choose font size based on image width
        font_size = (
            (1 - 2 * margin) * image_size[0] / (max_chars_per_line * char_width_ratio)
        )
    print(font_size)
    return max(1, font_size)


def text_symbol_to_mask(text, size=(64, 64), font_size="auto"):
    # Create image
    img = Image.new("L", size, 0)  # 'L' for grayscale
    draw = ImageDraw.Draw(img)

    if font_size == "auto":
        font_size = auto_fontsize(text, image_size=size)

    # Try to use a font, fallback to default
    try:
        font = ImageFont.truetype("FreeMonoBold.ttf", font_size)
    except Exception as e:
        print(e)
        font = ImageFont.load_default(font_size)

    draw.text(
        tuple(s // 2 for s in size),
        text,
        fill=256,
        font=font,
        anchor="mm",
        align="center",
        spacing=max(1, int(0.2 * font_size)),
    )
    return torch.tensor(img) > 128


def find_best_grid(N, H, W):
    aspect = H / W
    c_est = math.sqrt(N / aspect)

    best_rc = None
    best_score = float("inf")

    for c in range(max(1, int(c_est) - 5), int(c_est) + 6):
        r = math.ceil(N / c)
        if r * c >= N:
            # Try to make cells as square as possible: minimize distortion
            cell_aspect = (H / r) / (W / c)  # = (H * c) / (W * r)
            distortion = abs(
                math.log(cell_aspect)
            )  # log scale = distortion from square
            if distortion < best_score:
                best_score = distortion
                best_rc = (r, c)

    return best_rc


def best_point_grid(N, H, W):
    grid = torch.zeros((H, W))
    r, c = find_best_grid(N, H, W)
    h_step, h_offset = H // r, H % r
    w_step, w_offset = W // c, W % c
    for k, (i, j) in enumerate(itertools.product(range(r), range(c))):
        # if i * h_step + h_step // 2 < H and j * w_step + w_step // 2 < W:
        if k < N:
            grid[
                i * h_step + (h_step + h_offset) // 2,
                j * w_step + (w_step + w_offset) // 2,
            ] = 1
    return grid


def cartesian_outer_prod_nd(a1, a2):
    """
    Generate a cartesian product of two N-D arrays.
    Product is taken over first dimensions, other dimensions are concatenated.
    """
    assert a1.ndim == a2.ndim, "Both arrays must have the same number of dimensions."
    assert a1.ndim >= 2, "Input arrays must have at least two dimensions."
    a1 = a1.unsqueeze(0).repeat(a2.shape[0], *(1,) * a1.ndim).transpose(1, 0)
    a2 = a2.unsqueeze(0).repeat(a1.shape[0], *(1,) * a2.ndim)
    # print("outer prod. shapes: ", a1.shape, a2.shape)
    return torch.cat((a1, a2), dim=2).flatten(0, 1)


def dice_pos_mask(enc_size, spacing="thirds"):
    x = torch.zeros((6, enc_size, enc_size))
    s = enc_size
    mid = s // 2
    match spacing:
        case "thirds":
            lo = s // 3
            hi = 2 * s // 3
        case "quarters":
            lo = s // 4
            hi = 3 * s // 4
        case _:
            raise ValueError("Invalid spacing option. Use 'thirds' or 'quarters'.")

    # One
    x[0, mid, mid] = 1

    # Two
    x[1, lo, lo] = 1
    x[1, hi, hi] = 1

    # Three
    x[2, lo, lo] = 1
    x[2, mid, mid] = 1
    x[2, hi, hi] = 1

    # Four
    x[3, lo, lo] = 1
    x[3, lo, hi] = 1
    x[3, hi, lo] = 1
    x[3, hi, hi] = 1

    # Five
    x[4, lo, lo] = 1
    x[4, lo, hi] = 1
    x[4, mid, mid] = 1
    x[4, hi, lo] = 1
    x[4, hi, hi] = 1

    # Six
    x[5, lo, lo] = 1
    x[5, lo, hi] = 1
    x[5, mid, lo] = 1
    x[5, mid, hi] = 1
    x[5, hi, lo] = 1
    x[5, hi, hi] = 1
    return x
