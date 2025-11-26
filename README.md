# OpenCVStories

A simple interactive Python application for image and video processing using OpenCV. This project allows you to:

-   Apply filters (blur, sharpness, channel swap, grayscale, and color channel selection) to images and videos
-   Perform arithmetic operations (addition, subtraction, blending) between two images
-   Add stickers (with transparency) to images by clicking to position them
-   Record videos from your webcam and choose whether to save them
-   Play videos and upload images

## Features

### Filters

-   **Blur**: Apply Gaussian blur to images or videos
-   **Sharpness**: Apply a sharpening filter
-   **Channel Swap**: Swap R/B channels or show only a specific channel (R, G, B, or grayscale)
-   **Channel Selection**: Choose to apply filters to the whole image, grayscale, or a single color channel

### Arithmetic Operations

-   **Addition**: Add two images pixel-wise
-   **Subtraction**: Subtract one image from another pixel-wise
-   **Blending**: Blend two images with a custom alpha

### Stickers

-   Choose from up to 5 PNG stickers (with alpha channel) in the `stickers/` folder
-   Click on the image to position the sticker (multiple times if desired)
-   Press `s` to save, `r` to choose another sticker, or `q` to cancel

### Video

-   Record video from your webcam and choose to save or discard
-   Play video files from the `resources/` folder

## How to Use

1. **Install requirements**

    ```bash
    pip install opencv-python numpy
    ```

2. **Prepare folders**

    - Place your images and videos in the `resources/` folder
    - Place up to 5 PNG stickers (with transparency) in the `stickers/` folder

3. **Run the application**

    ```bash
    python main.py
    ```

4. **Follow the menu prompts**
    - Select channel mode (whole image, grayscale, R, G, or B)
    - Choose an operation (filter, arithmetic, sticker, etc.)
    - For stickers, click to position, then press `s` to save, `r` to restart, or `q` to cancel
    - For video recording, press `q` to stop and then choose to save or discard

## File Structure

```
main.py
resources/   # Place your images and videos here
stickers/    # Place your PNG stickers here (with alpha channel)
```

## Notes

-   Only PNG stickers with an alpha channel are supported
-   All changes can be saved or discarded after previewing
-   For arithmetic operations, images are resized to the smallest common size
-   The program is fully interactive via the console and OpenCV windows

## License

MIT License
