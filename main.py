from pathlib import Path
import shutil
import csv
import zipfile
import tempfile
import time
import resource

import gradio as gr
from PIL import Image, ImageDraw, UnidentifiedImageError
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


# =========================
# Settings
# =========================

MODEL_NAME = "facebook/dinov2-large"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def increase_open_file_limit(target=8192):
    """
    Large Gradio folder uploads can open many temporary files at once.
    Raising the soft limit helps avoid "Too many open files" on macOS/Linux.
    """
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft_limit = min(target, hard_limit)

        if soft_limit < new_soft_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
            print(f"Open file limit increased: {soft_limit} -> {new_soft_limit}")
    except (OSError, ValueError) as error:
        print(f"Could not increase open file limit: {error}")


increase_open_file_limit()


# =========================
# Device
# =========================

if torch.backends.mps.is_available():
    device = "mps"      # Apple Silicon Mac
elif torch.cuda.is_available():
    device = "cuda"     # NVIDIA GPU
else:
    device = "cpu"

print("Using device:", device)


# =========================
# Load DINOv2 once
# =========================

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


# =========================
# Helper functions
# =========================

def safe_copy(src_path, dst_dir):
    """
    Copy file into dst_dir while preserving the original filename.
    """
    src_path = Path(src_path)
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_path = dst_dir / src_path.name

    if dst_path.exists():
        raise gr.Error(
            "Duplicate output filename found: "
            f"{src_path.name}. Rename one source file or split the run."
        )

    shutil.copy2(src_path, dst_path)
    return dst_path


def make_zip(folder_path, zip_path):
    """
    Zip all files inside a folder.
    """
    folder_path = Path(folder_path)
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in folder_path.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(folder_path))

    return zip_path


def make_comparison_image(kept_path, removed_path, similarity, output_path):
    """
    Create a side-by-side preview image showing the kept reference and removed image.
    """
    kept_image = load_image(kept_path)
    removed_image = load_image(removed_path)

    if kept_image is None or removed_image is None:
        return None

    tile_size = (420, 420)
    label_height = 64
    padding = 16

    kept_image.thumbnail(tile_size, Image.Resampling.LANCZOS)
    removed_image.thumbnail(tile_size, Image.Resampling.LANCZOS)

    canvas_width = tile_size[0] * 2 + padding * 3
    canvas_height = tile_size[1] + label_height + padding * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    kept_x = padding
    removed_x = tile_size[0] + padding * 2
    image_y = label_height + padding

    canvas.paste(
        kept_image,
        (kept_x + (tile_size[0] - kept_image.width) // 2, image_y)
    )
    canvas.paste(
        removed_image,
        (removed_x + (tile_size[0] - removed_image.width) // 2, image_y)
    )

    draw = ImageDraw.Draw(canvas)
    draw.text((kept_x, padding), "KEPT", fill="green")
    draw.text((kept_x, padding + 22), Path(kept_path).name[:48], fill="black")
    draw.text((removed_x, padding), "REMOVED - SIMILAR", fill="red")
    draw.text((removed_x, padding + 22), Path(removed_path).name[:48], fill="black")
    draw.text(
        (removed_x, padding + 42),
        f"Similarity: {similarity:.4f}",
        fill="black"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def load_image(path):
    """
    Safely load an image as RGB.
    """
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def collect_image_paths(uploaded_files, local_folder_path):
    """
    Collect image paths from either Gradio uploads or a local folder path.
    Reading from a local path avoids Gradio upload limits for large batches.
    """
    if local_folder_path and local_folder_path.strip():
        folder_path = Path(local_folder_path.strip().strip("\"'")).expanduser()

        if not folder_path.exists():
            raise gr.Error(f"Local folder does not exist: {folder_path}")

        if not folder_path.is_dir():
            raise gr.Error(f"Local path is not a folder: {folder_path}")

        return sorted(
            path
            for path in folder_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not uploaded_files:
        return []

    return [
        Path(file)
        for file in uploaded_files
        if Path(file).suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_output_name_prefix(local_folder_path):
    """
    Use the selected local folder name as the ZIP filename prefix.
    """
    if local_folder_path and local_folder_path.strip():
        return Path(local_folder_path.strip().strip("\"'")).expanduser().name

    return "images"


def extract_embeddings(image_paths, batch_size, progress=gr.Progress()):
    """
    Extract DINOv2 embeddings for all images.
    Batch size only controls how many images are processed at once.
    All embeddings are compared later across the whole dataset.
    """
    all_embeddings = []
    valid_paths = []

    total_batches = max(1, (len(image_paths) + batch_size - 1) // batch_size)

    for batch_num, i in enumerate(range(0, len(image_paths), batch_size)):
        progress(
            batch_num / total_batches,
            desc=f"Extracting DINOv2 embeddings: batch {batch_num + 1}/{total_batches}"
        )

        batch_paths = image_paths[i:i + batch_size]
        images = []
        paths = []

        for path in batch_paths:
            image = load_image(path)
            if image is not None:
                images.append(image)
                paths.append(path)

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # CLS token = one feature vector for the whole image
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Normalize embeddings so dot product becomes cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())
        valid_paths.extend(paths)

    if not all_embeddings:
        return None, []

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings, valid_paths


# =========================
# Main DINOv2 similarity function
# =========================

def remove_similar_images(
    uploaded_files,
    local_folder_path,
    similarity_threshold_percent,
    batch_size,
    preview_limit,
    progress=gr.Progress()
):
    """
    uploaded_files: images uploaded through Gradio
    local_folder_path: local folder path for large image batches
    similarity_threshold_percent: 50 to 99 from the slider
    batch_size: how many images DINOv2 processes at once
    preview_limit: how many unique/similar images to show in each gallery
    """

    image_paths = collect_image_paths(uploaded_files, local_folder_path)

    if not image_paths:
        return (
            "No images found. Upload images or enter a local folder path.",
            None,
            None,
            None,
            [],
            [],
            []
        )

    # Convert percentage threshold to decimal
    # Example: 95 -> 0.95
    similarity_threshold = similarity_threshold_percent / 100

    if len(image_paths) == 0:
        return (
            "No valid image files found. Use JPG, JPEG, PNG, BMP, or WEBP.",
            None,
            None,
            None,
            [],
            [],
            []
        )

    timestamp = int(time.time())
    output_root = Path(tempfile.mkdtemp()) / f"dinov2_dedupe_{timestamp}"

    unique_dir = output_root / "unique_images"
    duplicate_dir = output_root / "similar_images"
    comparison_dir = output_root / "similarity_comparisons"

    unique_dir.mkdir(parents=True, exist_ok=True)
    duplicate_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    output_name_prefix = get_output_name_prefix(local_folder_path)
    report_path = output_root / "dedupe_report.csv"
    unique_zip_path = output_root / f"{output_name_prefix}_unique.zip"
    duplicate_zip_path = output_root / f"{output_name_prefix}_similar.zip"

    progress(0.05, desc="Preparing images")

    embeddings, valid_paths = extract_embeddings(
        image_paths=image_paths,
        batch_size=int(batch_size),
        progress=progress
    )

    if embeddings is None or len(valid_paths) == 0:
        return (
            "Could not read any uploaded images.",
            None,
            None,
            None,
            [],
            [],
            []
        )

    kept_indices = []
    duplicates = []

    progress(0.75, desc="Comparing image similarity")

    for idx in range(len(valid_paths)):
        current_embedding = embeddings[idx]

        if len(kept_indices) == 0:
            kept_indices.append(idx)
            continue

        kept_embeddings = embeddings[kept_indices]

        # Compare current image to all previously kept images
        similarities = torch.matmul(kept_embeddings, current_embedding)

        max_similarity, max_position = torch.max(similarities, dim=0)

        max_similarity = float(max_similarity.item())
        most_similar_kept_index = kept_indices[max_position.item()]

        if max_similarity >= similarity_threshold:
            duplicates.append({
                "duplicate_index": idx,
                "duplicate_image": valid_paths[idx],
                "kept_index": most_similar_kept_index,
                "kept_image": valid_paths[most_similar_kept_index],
                "similarity": max_similarity
            })
        else:
            kept_indices.append(idx)

    progress(0.85, desc="Copying files to output folders")

    # Copy unique images
    for idx in kept_indices:
        safe_copy(valid_paths[idx], unique_dir)

    # Copy duplicate/similar images
    for item in duplicates:
        safe_copy(item["duplicate_image"], duplicate_dir)

    # Write CSV report
    with open(report_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["similar_image", "kept_reference_image", "similarity"])

        for item in duplicates:
            writer.writerow([
                Path(item["duplicate_image"]).name,
                Path(item["kept_image"]).name,
                round(item["similarity"], 6)
            ])

    progress(0.95, desc="Creating zip files")

    make_zip(unique_dir, unique_zip_path)
    make_zip(duplicate_dir, duplicate_zip_path)

    # Side-by-side visual similarity report for Gradio
    comparison_gallery_items = []
    for count, item in enumerate(duplicates[:int(preview_limit)]):
        comparison_path = make_comparison_image(
            kept_path=item["kept_image"],
            removed_path=item["duplicate_image"],
            similarity=item["similarity"],
            output_path=comparison_dir / f"comparison_{count:05d}.jpg"
        )

        if comparison_path is not None:
            caption = (
                f"Kept: {Path(item['kept_image']).name} | "
                f"Removed: {Path(item['duplicate_image']).name} | "
                f"Similarity: {item['similarity']:.4f}"
            )
            comparison_gallery_items.append((str(comparison_path), caption))

    # Gallery preview of unique/kept images
    unique_gallery_items = []
    for idx in kept_indices[:int(preview_limit)]:
        image_path = valid_paths[idx]
        caption = f"Unique image: {Path(image_path).name}"
        unique_gallery_items.append((str(image_path), caption))

    # Gallery preview of similar/removed images
    duplicate_gallery_items = []
    for item in duplicates[:int(preview_limit)]:
        caption = (
            f"Similar to: {Path(item['kept_image']).name} | "
            f"Similarity: {item['similarity']:.4f}"
        )
        duplicate_gallery_items.append((str(item["duplicate_image"]), caption))

    summary = (
        f"DINOv2 similarity cleanup complete.\n\n"
        f"Model: {MODEL_NAME}\n"
        f"Device: {device}\n"
        f"Similarity threshold: {similarity_threshold_percent}% "
        f"({similarity_threshold:.2f})\n\n"
        f"Uploaded valid images: {len(valid_paths)}\n"
        f"Unique images kept: {len(kept_indices)}\n"
        f"Similar images found: {len(duplicates)}\n\n"
        f"Download unique_images.zip and similar_images.zip below."
    )

    progress(1.0, desc="Done")

    return (
        summary,
        str(unique_zip_path),
        str(duplicate_zip_path),
        str(report_path),
        comparison_gallery_items,
        unique_gallery_items,
        duplicate_gallery_items
    )


# =========================
# Gradio UI
# =========================

with gr.Blocks(title="DINOv2 Image Similarity Remover") as demo:
    gr.Markdown(
        """
        # DINOv2 Image Similarity Remover

        For large image sets, enter a local folder path below instead of using browser upload.
        Browser upload can fail when thousands of temporary files are opened at once.

        This app uses **Hugging Face DINOv2** embeddings and **cosine similarity**
        to separate unique images from very similar images.

        **Important:** This does not delete your original files.  
        It creates downloadable ZIP files.
        """
    )

    local_folder_path = gr.Textbox(
        label="Local folder path for large batches",
        placeholder="/Users/richard/Pictures/road_images",
        info="Recommended for large image sets. The app reads files directly from disk instead of uploading them through the browser."
    )

    uploaded_files = gr.File(
        label="Upload image folder for smaller batches",
        file_count="directory",
        file_types=["image"],
        type="filepath"
    )

    with gr.Row():
        similarity_threshold = gr.Slider(
            minimum=50,
            maximum=99,
            value=95,
            step=1,
            label="Similarity threshold (%)"
        )

        batch_size = gr.Slider(
            minimum=1,
            maximum=64,
            value=16,
            step=1,
            label="Batch size"
        )

        preview_limit = gr.Slider(
            minimum=5,
            maximum=100,
            value=30,
            step=5,
            label="Preview limit for each gallery"
        )

    run_button = gr.Button("Run DINOv2 Similarity Cleanup")

    summary_output = gr.Textbox(
        label="Summary",
        lines=10
    )

    with gr.Row():
        unique_zip_output = gr.File(label="Download unique images")
        duplicate_zip_output = gr.File(label="Download similar images")
        report_csv_output = gr.File(label="Download CSV report")

    comparison_gallery = gr.Gallery(
        label="Similarity report: kept image vs removed image",
        columns=1,
        height=600
    )

    unique_gallery = gr.Gallery(
        label="Preview of unique/kept images",
        columns=4,
        height=500
    )

    duplicate_gallery = gr.Gallery(
        label="Preview of similar/removed images",
        columns=4,
        height=500
    )

    run_button.click(
        fn=remove_similar_images,
        inputs=[
            uploaded_files,
            local_folder_path,
            similarity_threshold,
            batch_size,
            preview_limit
        ],
        outputs=[
            summary_output,
            unique_zip_output,
            duplicate_zip_output,
            report_csv_output,
            comparison_gallery,
            unique_gallery,
            duplicate_gallery
        ]
    )


demo.queue(default_concurrency_limit=1).launch(
    max_file_size="4GB",
    show_error=True
)