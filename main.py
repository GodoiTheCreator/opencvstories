import cv2
import numpy as np
import os

def upload_image(file_path):
    """Uploads an image from the given file path."""
    image = cv2.imread(file_path)
    cv2.imshow("Uploaded Image", image)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    return image

def record_video():
    video = cv2.VideoCapture(0) # Captures video from the default camera
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # four-character code used to identify data formats, in this case video codec
    out = cv2.VideoWriter('resources/video.mp4', fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything if job is finished
    video.release()
    out.release()
    cv2.destroyAllWindows()

def play_video(file_path):
    video = cv2.VideoCapture(file_path)
    while True:
        success, img = video.read()
        # The images will be saved in img, and success will tell us if this worked, as a boolean variable, true or false
        cv2.imshow("Video", img)
        # This show all images
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # If we press q, the video will close, if not, the video continues until his end
            break
        
def save_edited_image(image, name="edited_image.jpg"):
    cv2.imwrite('resources/' + name, image)
    print(f"Edited image saved to {file_path}")

#def apply_filter_blur(file_path):
#    image = cv2.imread(file_path)
#    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
#    cv2.imshow("Blurred Image", blurred_image)
#    return blurred_image

#def apply_filter_sharpness(file_path):
#    image = cv2.imread(file_path)
#    kernel = np.array([[0, -1, 0], 
#                       [-1, 5,-1], 
#                       [0, -1, 0]])
#    sharp_image = cv2.filter2D(image, -1, kernel)
#    cv2.imshow("Sharpness Filtered Image", sharp_image)
#    return sharp_image

#def apply_channel_swap(file_path):
#    image = cv2.imread(file_path)
#    swapped_image = image.copy()
#    swapped_image[:, :, [0, 2]] = swapped_image[:, :, [2, 0]]  # Swap Blue and Red channels
#    cv2.imshow("Channel Swapped Image", swapped_image)
#    return swapped_image

def detect_file_type(file_path):
    img = cv2.imread(file_path)
    if img is not None:
        return "image"
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
        cap.release()
        return "video"
    cap.release()
    return "unknown"

def apply_filter_blur(file_path):
    file_type = detect_file_type(file_path)
    if file_type == "image":
        image = cv2.imread(file_path)
        blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
        cv2.imshow("Blurred Image", blurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif file_type == "video":
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
            cv2.imshow("Blurred Video", blurred_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unsupported file type.")
        
def apply_filter_sharpness(file_path):
    file_type = detect_file_type(file_path)
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    if file_type == "image":
        image = cv2.imread(file_path)
        sharp_image = cv2.filter2D(image, -1, kernel)
        cv2.imshow(f"Sharpened Image - {os.path.basename(file_path)}", sharp_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif file_type == "video":
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            sharp_frame = cv2.filter2D(frame, -1, kernel)
            cv2.imshow(f"Sharpened Video - {os.path.basename(file_path)}", sharp_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Unsupported file type for {file_path}.")
        
def apply_channel_swapl(file_path):
    file_type = detect_file_type(file_path)
    if file_type == "image":
        image = cv2.imread(file_path)
        print("Channel options:")
        print("1. Apply to entire image (swap R and B)")
        print("2. Show only Red channel")
        print("3. Show only Green channel")
        print("4. Show only Blue channel")
        print("5. Show only Grey Scale channel")
        channel_choice = input("Choose an option (1-5): ")
        if channel_choice == '1':
            swapped_image = image.copy()
            swapped_image[:, :, [0, 2]] = swapped_image[:, :, [2, 0]]
            cv2.imshow(f"Channel Swapped Image - {os.path.basename(file_path)}", swapped_image)
        elif channel_choice == '2':
            red_channel = image[:, :, 2]
            red_img = np.zeros_like(image)
            red_img[:, :, 2] = red_channel
            cv2.imshow(f"Red Channel - {os.path.basename(file_path)}", red_img)
        elif channel_choice == '3':
            green_channel = image[:, :, 1]
            green_img = np.zeros_like(image)
            green_img[:, :, 1] = green_channel
            cv2.imshow(f"Green Channel - {os.path.basename(file_path)}", green_img)
        elif channel_choice == '4':
            blue_channel = image[:, :, 0]
            blue_img = np.zeros_like(image)
            blue_img[:, :, 0] = blue_channel
            cv2.imshow(f"Blue Channel - {os.path.basename(file_path)}", blue_img)
        elif channel_choice == '6':
            grey_img = cv2.cvtColor(grey_channel, cv2.COLOR_BGR2GRAY)
            cv2.imshow(f"Grey Scale Channel - {os.path.basename(file_path)}", grey_img)
        else:
            print("Invalid channel option.")
            return
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif file_type == "video":
        print("Channel options:")
        print("1. Apply to entire video (swap R and B)")
        print("2. Show only Red channel")
        print("3. Show only Green channel")
        print("4. Show only Blue channel")
        channel_choice = input("Choose an option (1-4): ")
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if channel_choice == '1':
                swapped_frame = frame.copy()
                swapped_frame[:, :, [0, 2]] = swapped_frame[:, :, [2, 0]]
                cv2.imshow(f"Channel Swapped Video - {os.path.basename(file_path)}", swapped_frame)
            elif channel_choice == '2':
                red_channel = frame[:, :, 2]
                red_img = np.zeros_like(frame)
                red_img[:, :, 2] = red_channel
                cv2.imshow(f"Red Channel Video - {os.path.basename(file_path)}", red_img)
            elif channel_choice == '3':
                green_channel = frame[:, :, 1]
                green_img = np.zeros_like(frame)
                green_img[:, :, 1] = green_channel
                cv2.imshow(f"Green Channel Video - {os.path.basename(file_path)}", green_img)
            elif channel_choice == '4':
                blue_channel = frame[:, :, 0]
                blue_img = np.zeros_like(frame)
                blue_img[:, :, 0] = blue_channel
                cv2.imshow(f"Blue Channel Video - {os.path.basename(file_path)}", blue_img)
            else:
                print("Invalid channel option.")
                break
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Unsupported file type for {file_path}.")

def apply_addition(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("One or both images could not be loaded.")
        return
    # Resize to smallest common size
    min_shape = (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1]))
    img1_resized = cv2.resize(img1, (min_shape[1], min_shape[0]))
    img2_resized = cv2.resize(img2, (min_shape[1], min_shape[0]))
    added = cv2.add(img1_resized, img2_resized)
    cv2.imshow("Addition Result", added)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_subtract(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("One or both images could not be loaded.")
        return
    min_shape = (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1]))
    img1_resized = cv2.resize(img1, (min_shape[1], min_shape[0]))
    img2_resized = cv2.resize(img2, (min_shape[1], min_shape[0]))
    subtracted = cv2.subtract(img1_resized, img2_resized)
    cv2.imshow("Subtraction Result", subtracted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_blending(img1_path, img2_path, alpha=0.5):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("One or both images could not be loaded.")
        return
    min_shape = (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1]))
    img1_resized = cv2.resize(img1, (min_shape[1], min_shape[0]))
    img2_resized = cv2.resize(img2, (min_shape[1], min_shape[0]))
    beta = 1.0 - alpha
    blended = cv2.addWeighted(img1_resized, alpha, img2_resized, beta, 0)
    cv2.imshow("Blending Result", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

while True:
    print("Select an option:")
    print("1. Upload an image")
    print("2. Record a video from webcam")
    print("3. Play a video from file")
    print("4. Apply blur filter to uploaded image")
    print("5. Apply sharpness filter to uploaded image")
    print("6. Apply channel swap filter to uploaded image")
    print("7. Addition of two images")
    print("8. Subtraction of two images")
    print("9. Blending of two images")
    print("10. Exit")
    
    choice = input("Enter your choice (1-10): ")
    if choice == '1':
        image_path = input("Enter the image file path: ")
        uploaded_image = upload_image(image_path)
    elif choice == '2':
        record_video()
    elif choice == '3':
        print("Choose a video to play:")
        for file in os.listdir("resources/"):
            if file.endswith(".mp4"):
                print(file)
        video_file = input("Enter the video file name: ")
        play_video(os.path.join("resources/", video_file))
    elif choice == '4':
        files = [file for file in os.listdir("resources/") if file.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv"))]
        if files:
            print("Available files for blur filter:")
            for file in files:
                print(file)
            selected_file = input("Enter the file name to apply blur filter: ")
            file_path = os.path.join("resources/", selected_file)
            if os.path.exists(file_path):
                apply_filter_blur(file_path)
            else:
                print("File not found.")
        else:
            print("No images or videos found in resources folder.")
    elif choice == '5':
        files = [file for file in os.listdir("resources/") if file.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv"))]
        if files:
            print("Available files for sharpness filter:")
            for file in files:
                print(file)
            selected_file = input("Enter the file name to apply sharpness filter: ")
            file_path = os.path.join("resources/", selected_file)
            if os.path.exists(file_path):
                apply_filter_sharpness(file_path)
            else:
                print("File not found.")
        else:
            print("No images or videos found in resources folder.")
    elif choice == '6':
        files = [file for file in os.listdir("resources/") if file.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv"))]
        if files:
            print("Available files for channel swap/filter:")
            for file in files:
                print(file)
            selected_file = input("Enter the file name to apply channel swap/filter: ")
            file_path = os.path.join("resources/", selected_file)
            if os.path.exists(file_path):
                apply_channel_swapl(file_path)
            else:
                print("File not found.")
        else:
            print("No images or videos found in resources folder.")
    elif choice == '7':
        # Addition
        files = [file for file in os.listdir("resources/") if file.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(files) >= 2:
            print("Available images for addition:")
            for file in files:
                print(file)
            img1 = input("Enter the first image file name: ")
            img2 = input("Enter the second image file name: ")
            img1_path = os.path.join("resources/", img1)
            img2_path = os.path.join("resources/", img2)
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                apply_addition(img1_path, img2_path)
            else:
                print("One or both files not found.")
        else:
            print("Not enough images in resources folder.")
    elif choice == '8':
        # Subtraction
        files = [file for file in os.listdir("resources/") if file.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(files) >= 2:
            print("Available images for subtraction:")
            for file in files:
                print(file)
            img1 = input("Enter the first image file name: ")
            img2 = input("Enter the second image file name: ")
            img1_path = os.path.join("resources/", img1)
            img2_path = os.path.join("resources/", img2)
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                apply_subtract(img1_path, img2_path)
            else:
                print("One or both files not found.")
        else:
            print("Not enough images in resources folder.")
    elif choice == '9':
        # Blending
        files = [file for file in os.listdir("resources/") if file.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(files) >= 2:
            print("Available images for blending:")
            for file in files:
                print(file)
            img1 = input("Enter the first image file name: ")
            img2 = input("Enter the second image file name: ")
            try:
                alpha = float(input("Enter alpha value for blending (0.0 to 1.0, default 0.5): ") or "0.5")
            except ValueError:
                alpha = 0.5
            img1_path = os.path.join("resources/", img1)
            img2_path = os.path.join("resources/", img2)
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                apply_blending(img1_path, img2_path, alpha)
            else:
                print("One or both files not found.")
        else:
            print("Not enough images in resources folder.")
    elif choice == '10':
        print("Exiting the program.")
        break
    else:
        print("Invalid option. Please enter a number between 1 and 10.")
