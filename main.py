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
        swapped_image = image.copy()
        swapped_image[:, :, [0, 2]] = swapped_image[:, :, [2, 0]]
        cv2.imshow(f"Channel Swapped Image - {os.path.basename(file_path)}", swapped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif file_type == "video":
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            swapped_frame = frame.copy()
            swapped_frame[:, :, [0, 2]] = swapped_frame[:, :, [2, 0]]
            cv2.imshow(f"Channel Swapped Video - {os.path.basename(file_path)}", swapped_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Unsupported file type for {file_path}.")



while True:
    print("Select an option:")
    print("1. Upload an image")
    print("2. Record a video from webcam")
    print("3. Play a video from file")
    print("4. Apply blur filter to uploaded image")
    print("5. Apply sharpness filter to uploaded image")
    print("6. Apply channel swap filter to uploaded image")
    print("7. Exit")
    
    choice = input("Enter your choice (1-7): ")
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
            print("Available files for channel swap filter:")
            for file in files:
                print(file)
            selected_file = input("Enter the file name to apply channel swap filter: ")
            file_path = os.path.join("resources/", selected_file)
            if os.path.exists(file_path):
                apply_channel_swapl(file_path)
            else:
                print("File not found.")
        else:
            print("No images or videos found in resources folder.")
    elif choice == '7':
        print("Exiting the program.")
        break
    else:
        print("Invalid option. Please enter a number between 1 and 7.")
