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
        
def save_edited_image(image, file_path):
    cv2.imwrite(file_path, image)
    print(f"Edited image saved to {file_path}")

def save_edited_video(frames, file_path, fps=30):
    if not frames:
        print("No frames to save.")
        return
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Edited video saved to {file_path}")

def ask_save_and_overwrite(image_or_frames, file_path, is_video=False, fps=30):
    save = input("Deseja salvar a alteração e sobrescrever o arquivo original? (s/n): ").strip().lower()
    if save == 's':
        if is_video:
            save_edited_video(image_or_frames, file_path, fps)
        else:
            save_edited_image(image_or_frames, file_path)
    else:
        print("Alteração descartada.")

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

def show_channel(image, channel):
    if channel == 'r':
        red_img = np.zeros_like(image)
        red_img[:, :, 2] = image[:, :, 2]
        return red_img
    elif channel == 'g':
        green_img = np.zeros_like(image)
        green_img[:, :, 1] = image[:, :, 1]
        return green_img
    elif channel == 'b':
        blue_img = np.zeros_like(image)
        blue_img[:, :, 0] = image[:, :, 0]
        return blue_img
    elif channel == 'gray':
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    else:
        return image

def apply_filter_blur(file_path, channel_mode):
    file_type = detect_file_type(file_path)
    if file_type == "image":
        image = cv2.imread(file_path)
        if channel_mode in ['r', 'g', 'b', 'gray']:
            image = show_channel(image, channel_mode)
        blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
        cv2.imshow("Blurred Image", blurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ask_save_and_overwrite(blurred_image, file_path)
    elif file_type == "video":
        cap = cv2.VideoCapture(file_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if channel_mode in ['r', 'g', 'b', 'gray']:
                frame = show_channel(frame, channel_mode)
            blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
            frames.append(blurred_frame)
            cv2.imshow("Blurred Video", blurred_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        ask_save_and_overwrite(frames, file_path, is_video=True, fps=fps)
    else:
        print("Unsupported file type.")

def apply_filter_sharpness(file_path, channel_mode):
    file_type = detect_file_type(file_path)
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    if file_type == "image":
        image = cv2.imread(file_path)
        if channel_mode in ['r', 'g', 'b', 'gray']:
            image = show_channel(image, channel_mode)
        sharp_image = cv2.filter2D(image, -1, kernel)
        cv2.imshow(f"Sharpened Image - {os.path.basename(file_path)}", sharp_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ask_save_and_overwrite(sharp_image, file_path)
    elif file_type == "video":
        cap = cv2.VideoCapture(file_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if channel_mode in ['r', 'g', 'b', 'gray']:
                frame = show_channel(frame, channel_mode)
            sharp_frame = cv2.filter2D(frame, -1, kernel)
            frames.append(sharp_frame)
            cv2.imshow(f"Sharpened Video - {os.path.basename(file_path)}", sharp_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        ask_save_and_overwrite(frames, file_path, is_video=True, fps=fps)
    else:
        print(f"Unsupported file type for {file_path}.")

def apply_channel_swapl(file_path, channel_mode):
    file_type = detect_file_type(file_path)
    if file_type == "image":
        image = cv2.imread(file_path)
        if channel_mode == 'swap':
            swapped_image = image.copy()
            swapped_image[:, :, [0, 2]] = swapped_image[:, :, [2, 0]]
            cv2.imshow(f"Channel Swapped Image - {os.path.basename(file_path)}", swapped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ask_save_and_overwrite(swapped_image, file_path)
        elif channel_mode in ['r', 'g', 'b', 'gray']:
            channel_img = show_channel(image, channel_mode)
            cv2.imshow(f"Channel {channel_mode.upper()} - {os.path.basename(file_path)}", channel_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ask_save_and_overwrite(channel_img, file_path)
        else:
            print("Invalid channel option.")
            return
    elif file_type == "video":
        cap = cv2.VideoCapture(file_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if channel_mode == 'swap':
                swapped_frame = frame.copy()
                swapped_frame[:, :, [0, 2]] = swapped_frame[:, :, [2, 0]]
                frames.append(swapped_frame)
                cv2.imshow(f"Channel Swapped Video - {os.path.basename(file_path)}", swapped_frame)
            elif channel_mode in ['r', 'g', 'b', 'gray']:
                channel_img = show_channel(frame, channel_mode)
                frames.append(channel_img)
                cv2.imshow(f"Channel {channel_mode.upper()} Video - {os.path.basename(file_path)}", channel_img)
            else:
                print("Invalid channel option.")
                break
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        ask_save_and_overwrite(frames, file_path, is_video=True, fps=fps)
    else:
        print(f"Unsupported file type for {file_path}.")

def apply_addition(img1_path, img2_path, channel_mode):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("One or both images could not be loaded.")
        return
    min_shape = (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1]))
    img1_resized = cv2.resize(img1, (min_shape[1], min_shape[0]))
    img2_resized = cv2.resize(img2, (min_shape[1], min_shape[0]))
    if channel_mode in ['r', 'g', 'b', 'gray']:
        img1_resized = show_channel(img1_resized, channel_mode)
        img2_resized = show_channel(img2_resized, channel_mode)
    added = cv2.add(img1_resized, img2_resized)
    cv2.imshow("Addition Result", added)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ask_save_and_overwrite(added, img1_path)

def apply_subtract(img1_path, img2_path, channel_mode):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("One or both images could not be loaded.")
        return
    min_shape = (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1]))
    img1_resized = cv2.resize(img1, (min_shape[1], min_shape[0]))
    img2_resized = cv2.resize(img2, (min_shape[1], min_shape[0]))
    if channel_mode in ['r', 'g', 'b', 'gray']:
        img1_resized = show_channel(img1_resized, channel_mode)
        img2_resized = show_channel(img2_resized, channel_mode)
    subtracted = cv2.subtract(img1_resized, img2_resized)
    cv2.imshow("Subtraction Result", subtracted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ask_save_and_overwrite(subtracted, img1_path)

def apply_blending(img1_path, img2_path, alpha=0.5, channel_mode='all'):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("One or both images could not be loaded.")
        return
    min_shape = (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1]))
    img1_resized = cv2.resize(img1, (min_shape[1], min_shape[0]))
    img2_resized = cv2.resize(img2, (min_shape[1], min_shape[0]))
    if channel_mode in ['r', 'g', 'b', 'gray']:
        img1_resized = show_channel(img1_resized, channel_mode)
        img2_resized = show_channel(img2_resized, channel_mode)
    beta = 1.0 - alpha
    blended = cv2.addWeighted(img1_resized, alpha, img2_resized, beta, 0)
    cv2.imshow("Blending Result", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ask_save_and_overwrite(blended, img1_path)

def add_sticker_to_image(image_path):
    stickers_dir = "stickers"
    stickers = [f for f in os.listdir(stickers_dir) if f.lower().endswith(('.png'))]
    if len(stickers) == 0:
        print("No stickers found in the stickers folder.")
        return

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Image not found.")
        return

    # If image has no alpha, add one for easier overlay
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    sticker_img = None
    sticker_name = None
    preview_img = img.copy()
    pos = (0, 0)

    def overlay_sticker(base_img, sticker, pos):
        x, y = pos
        h, w = sticker.shape[:2]
        overlay = base_img.copy()
        # Ensure sticker fits in image
        if y + h > overlay.shape[0] or x + w > overlay.shape[1]:
            return overlay
        alpha_s = sticker[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            overlay[y:y+h, x:x+w, c] = (alpha_s * sticker[:, :, c] +
                                        alpha_l * overlay[y:y+h, x:x+w, c])
        return overlay

    def mouse_callback(event, x, y, flags, param):
        nonlocal preview_img, pos
        if event == cv2.EVENT_LBUTTONDOWN and sticker_img is not None:
            pos = (x, y)
            preview_img = overlay_sticker(img, sticker_img, (x, y))
            cv2.imshow("Add Sticker", preview_img)

    while True:
        print("\nEscolha um sticker para adicionar:")
        for idx, s in enumerate(stickers[:5]):
            print(f"{idx+1}. {s}")
        print("0. Cancelar edição")
        try:
            choice = int(input("Digite o número do sticker (1-5) ou 0 para cancelar: "))
        except ValueError:
            print("Escolha inválida.")
            continue
        if choice == 0:
            print("Edição de sticker cancelada.")
            return
        if 1 <= choice <= min(5, len(stickers)):
            sticker_name = stickers[choice-1]
            sticker_img = cv2.imread(os.path.join(stickers_dir, sticker_name), cv2.IMREAD_UNCHANGED)
            if sticker_img is None or sticker_img.shape[2] != 4:
                print("Sticker inválido (precisa ter canal alfa).")
                continue
            break
        else:
            print("Escolha inválida.")

    print("Clique na imagem para posicionar o sticker. Pressione 's' para salvar, 'r' para escolher outro sticker, ou 'q' para cancelar.")
    preview_img = img.copy()
    cv2.namedWindow("Add Sticker")
    cv2.setMouseCallback("Add Sticker", mouse_callback)
    cv2.imshow("Add Sticker", preview_img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Salvar imagem editada
            cv2.destroyAllWindows()
            save_edited_image(preview_img, image_path)
            break
        elif key == ord('r'):
            # Reescolher sticker
            cv2.destroyAllWindows()
            add_sticker_to_image(image_path)
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("Edição de sticker cancelada.")
            break

while True:
    print("\nSelecione o modo de canal para aplicar os filtros:")
    print("1. Imagem inteira (RGB)")
    print("2. Escala de cinza (Grayscale)")
    print("3. Apenas canal Vermelho (R)")
    print("4. Apenas canal Verde (G)")
    print("5. Apenas canal Azul (B)")
    channel_option = input("Escolha uma opção (1-5): ")

    if channel_option == '1':
        channel_mode = 'all'
    elif channel_option == '2':
        channel_mode = 'gray'
    elif channel_option == '3':
        channel_mode = 'r'
    elif channel_option == '4':
        channel_mode = 'g'
    elif channel_option == '5':
        channel_mode = 'b'
    else:
        print("Opção inválida.")
        continue

    print("\nSelecione uma operação:")
    print("1. Upload an image")
    print("2. Record a video from webcam")
    print("3. Play a video from file")
    print("4. Apply blur filter")
    print("5. Apply sharpness filter")
    print("6. Apply 5channel swap filter (swap R/B ou mostrar canal)")
    print("7. Addition of two images")
    print("8. Subtraction of two images")
    print("9. Blending of two images")
    print("10. Exit")
    print("11. Adicionar sticker a uma imagem")

    choice = input("Enter your choice (1-11): ")
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
                apply_filter_blur(file_path, channel_mode)
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
                apply_filter_sharpness(file_path, channel_mode)
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
                if channel_mode == 'all':
                    apply_channel_swapl(file_path, 'swap')
                else:
                    apply_channel_swapl(file_path, channel_mode)
            else:
                print("File not found.")
        else:
            print("No images or videos found in resources folder.")
    elif choice == '7':
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
                apply_addition(img1_path, img2_path, channel_mode)
            else:
                print("One or both files not found.")
        else:
            print("Not enough images in resources folder.")
    elif choice == '8':
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
                apply_subtract(img1_path, img2_path, channel_mode)
            else:
                print("One or both files not found.")
        else:
            print("Not enough images in resources folder.")
    elif choice == '9':
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
                apply_blending(img1_path, img2_path, alpha, channel_mode)
            else:
                print("One or both files not found.")
        else:
            print("Not enough images in resources folder.")
    elif choice == '10':
        print("Exiting the program.")
        break
    elif choice == '11':
        files = [file for file in os.listdir("resources/") if file.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            print("Imagens disponíveis para adicionar sticker:")
            for file in files:
                print(file)
            selected_file = input("Digite o nome da imagem para adicionar sticker: ")
            file_path = os.path.join("resources/", selected_file)
            if os.path.exists(file_path):
                add_sticker_to_image(file_path)
            else:
                print("Arquivo não encontrado.")
        else:
            print("Nenhuma imagem encontrada na pasta resources.")
    else:
        print("Invalid option. Please enter a number between 1 and 11.")
