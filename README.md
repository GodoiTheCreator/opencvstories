# OpenCV Stories

OpenCV Stories is a Python application for image and video processing, offering a suite of filters, effects, and editing tools. It features both a command-line interface (CLI) and a user-friendly Graphical User Interface (GUI) built with Tkinter.

## 📋 Project Overview

This project demonstrates the power of OpenCV for real-time media manipulation. Users can upload images, record videos, apply various filters (blur, sharpen, channel swap), perform arithmetic operations on images, and add stickers.

## 📂 Project Structure

```
opencvstories/
├── main.py              # Core logic and CLI implementation
├── gui.py               # Graphical User Interface (Tkinter)
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
├── resources/           # Directory for saving/loading media
└── stickers/            # Directory for sticker images (PNGs)
```

## 🛠 Installation

1.  **Prerequisites**: Ensure Python 3.x is installed.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: On macOS, you may need to install `python-tk` via Homebrew if you encounter Tkinter issues.*

## 🚀 Usage

### Graphical User Interface (Recommended)
Run the GUI to access all features in a windowed environment:
```bash
python3 gui.py
```

**Features:**
*   **File Operations**: Load images/videos or record from webcam.
*   **Channel Modes**: View specific color channels (RGB, Gray, Red, Green, Blue).
*   **Filters**: Apply Blur, Sharpen, or Channel Swap effects.
*   **Stickers**: Interactively place stickers on images.
*   **Arithmetic**: Add, Subtract, or Blend two images together.

### Command Line Interface
Run the script directly to use the interactive menu in the terminal:
```bash
python3 main.py
```

## 💻 Code Documentation

### `main.py`
Contains the core image processing functions.

**Key Functions:**
*   `upload_image(file_path)`: Loads an image from disk.
*   `record_video(interactive=True)`: Captures video from the webcam.
*   `apply_filter_blur(file_path, channel_mode)`: Applies Gaussian blur.
*   `apply_filter_sharpness(file_path, channel_mode)`: Applies a sharpening kernel.
*   `overlay_sticker(base_img, sticker, pos)`: Overlays a transparent PNG sticker onto a base image at a specific position, handling clipping and transparency.
*   `apply_blending(...)`: Blends two images with a specified alpha value.

### `gui.py`
Implements the Tkinter GUI.

**Class: `OpenCVStoriesApp`**
*   **`__init__`**: Initializes the main window and widgets.
*   **`create_widgets`**: Sets up the layout (buttons, canvas, status bar).
*   **`show_image_window`**: Helper method to display processed images in a separate window with "Save" and "Close" options.
*   **`add_sticker`**: Opens a dedicated window for interactive sticker placement using mouse clicks.
*   **`preview_file`**: Renders the current image or video frame on the main canvas.

## 🎨 GUI Details

The GUI is designed for simplicity:
*   **Control Panel**: Top section containing buttons for file loading, recording, and channel selection.
*   **Action Bar**: Buttons for applying filters and effects.
*   **Canvas**: Central area displaying the currently loaded media.
*   **Status Bar**: Bottom bar showing current status and instructions.

When applying effects, a new window opens to show the result, allowing you to compare with the original before saving.
