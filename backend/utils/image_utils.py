from PIL import Image, ImageChops, ImageEnhance
import os
import tempfile

def save_ela_image(img, original_path):
    """
    Basic ELA implementation:
    - save re-saved image at lower quality
    - compute absolute difference
    - enhance and save image
    Returns (ela_path, ela_score)
    """
    base = os.path.splitext(os.path.basename(original_path))[0]
    tmpname = f"{base}_ela.png"
    tmpdir = os.path.join(os.path.dirname(original_path), "ela")
    os.makedirs(tmpdir, exist_ok=True)
    ela_out = os.path.join(tmpdir, tmpname)

    # save compressed copy to temporary buffer
    tmpfile = os.path.join(tmpdir, "tmp_resave.jpg")
    img.save(tmpfile, "JPEG", quality=90)
    reloaded = Image.open(tmpfile).convert("RGB")
    diff = ImageChops.difference(img, reloaded)
    # amplify difference
    extrema = diff.getextrema()
    max_diff = max([e[1] for e in extrema])
    if max_diff == 0:
        scale = 1
    else:
        scale = 255.0 / max_diff
    enhanced = ImageEnhance.Brightness(diff).enhance(scale)
    enhanced.save(ela_out)

    # crude ela_score: mean pixel value
    stat = enhanced.convert("L").getextrema()
    ela_score = enhanced.convert("L").histogram()
    # quick summary: average brightness
    pixels = enhanced.convert("L").getdata()
    avg = sum(pixels) / max(1, len(pixels))
    return ela_out, avg
